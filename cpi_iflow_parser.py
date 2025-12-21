#!/usr/bin/env python3
"""
SAP CPI AI Explainer - Batch Processor
1. Processes multiple ZIP files provided as arguments.
2. Uses DeepSeek API for step-by-step analysis and architectural summary.
3. Outputs: iflow_analysis_for_<filename>.json for each input.
"""

import zipfile
import xml.etree.ElementTree as ET
import sys
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

# --- CONFIGURATION ---
# The dummy key provided in the request
DEEPSEEK_API_KEY = "sk-f7d101c4a97246318ab270f5d67abfdd"
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
MAX_STEPS = 500  # Safety limit for steps per iFlow

NAMESPACES = {
    'bpmn2': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
    'ifl': 'http:///com.sap.ifl.model/Ifl.xsd',
}
for prefix, uri in NAMESPACES.items():
    ET.register_namespace(prefix, uri)

# --- PROMPTS ---
ARCHITECT_SYSTEM_PROMPT = """You are an SAP CPI Integration Architect.
Your task is to generate a HIGH-QUALITY INTEGRATION DESCRIPTION based on all the steps we analyzed.
This will be used for semantic search and retrieval (RAG).

STRICT RULES:
- Output ONLY the description.
- Do NOT list steps or phases.
- Do NOT explain implementation details.
- Do NOT speculate or add business justification.
- Do NOT use generic phrases like "various processing".

WHAT TO INCLUDE:
- Source and Target system(s).
- Type of data exchanged.
- Key technical behaviors (scheduled, event-driven, routing, transformation).

STYLE:
- 2–4 concise sentences.
- , concrete, and factual.
- Plain text only."""

class DeepSeekAIHelper:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = API_URL

    def _call_api(self, messages):
        payload = {
            "model": MODEL,
            "messages": messages,
            "stream": False
        }
        
        # Exponential Backoff Implementation
        max_retries = 5
        for i in range(max_retries):
            try:
                req = urllib.request.Request(
                    self.url,
                    data=json.dumps(payload).encode('utf-8'),
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {self.api_key}'
                    }
                )
                with urllib.request.urlopen(req) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    return result["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as e:
                if e.code == 429:  # Rate limit
                    wait = 2 ** i
                    time.sleep(wait)
                    continue
                return f"(API Error: {e.code} - {e.reason})"
            except Exception as e:
                return f"(Connection Error: {str(e)})"
        return "(Error: Max retries exceeded)"

    def get_step_summary(self, context, data):
        if not data: return "Standard processing step."

        system_msg = "You are a concise SAP CPI expert. Summarize this step and tell what we did exactly in this step with in 1-3 lines. u can mention any fields/variables/or any we used in this in summary."
        user_msg = f"Explain this '{context}' step. TECHNICAL DATA:\n{data}"
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        return self._call_api(messages)

    def get_architectural_analysis(self, full_flow_text):
        messages = [
            {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze the following SAP CPI iFlow steps and provide the architectural description:\n\n{full_flow_text}"}
        ]
        return self._call_api(messages)

class CPIFlowAnalyzer:
    def __init__(self, zip_path, ai_helper):
        self.zip_path = Path(zip_path)
        self.ai = ai_helper
        self.xml_content = None
        self.root = None
        self.elements = {}
        self.adapter_configs = {} 
        self.parameters = {}       
        self.step_counter = 0
        self.zip_file = None
        self.rag_data = {
            "iflow_file": self.zip_path.name,
            "steps": [],
            "architectural_analysis": ""
        }

    def load(self):
        try:
            self.zip_file = zipfile.ZipFile(self.zip_path, 'r')
            self._load_parameters()
            # Find the core integration flow XML
            candidates = [f for f in self.zip_file.namelist() if f.endswith('.iflw') and 'integrationflow' in f.lower()]
            if not candidates:
                print(f"  [!] No .iflw file found in {self.zip_path.name}")
                return False
            with self.zip_file.open(candidates[0]) as f:
                self.xml_content = f.read()
            self.root = ET.fromstring(self.xml_content)
            self._parse_elements()
            return True
        except Exception as e:
            print(f"  [!] Error loading {self.zip_path.name}: {e}")
            return False

    def _load_parameters(self):
        try:
            candidates = [f for f in self.zip_file.namelist() if f.endswith('parameters.prop')]
            if not candidates: return
            with self.zip_file.open(candidates[0]) as f:
                content = f.read().decode('utf-8', errors='ignore')
                for line in content.splitlines():
                    if '=' in line and not line.startswith('#'):
                        k, v = line.split('=', 1)
                        self.parameters[k.strip()] = v.strip()
        except: pass

    def _resolve_placeholders(self, props):
        res = {}
        for k, v in props.items():
            if isinstance(v, str) and '{{' in v:
                clean = v.replace('{{', '').replace('}}', '').strip()
                res[k] = self.parameters.get(clean, v)
            else:
                res[k] = v
        return res

    def _parse_elements(self):
        # Parse Message Flows (Adapters)
        for mf in self.root.findall('.//bpmn2:messageFlow', NAMESPACES):
            props = self._get_ifl_properties(mf)
            props = self._resolve_placeholders(props)
            s, t = mf.get('sourceRef'), mf.get('targetRef')
            if s: self.adapter_configs[s] = props
            if t: self.adapter_configs[t] = props

        # Parse BPMN Elements
        for elem in self.root.findall('.//*[@id]'):
            eid = elem.get('id')
            tag = elem.tag.split('}')[-1]
            props = self._get_ifl_properties(elem)
            props = self._resolve_placeholders(props)
            self.elements[eid] = {
                'id': eid,
                'name': elem.get('name', ''),
                'tag': tag,
                'type': self._determine_readable_type(elem, tag, props),
                'props': props,
                'outgoing': []
            }
            
        # Map connections
        for sf in self.root.findall('.//bpmn2:sequenceFlow', NAMESPACES):
            s, t = sf.get('sourceRef'), sf.get('targetRef')
            if s in self.elements and t in self.elements:
                self.elements[s]['outgoing'].append({'target': t})

    def _get_ifl_properties(self, elem):
        props = {}
        for child in elem.iter():
            if child.tag.endswith('property'):
                k, v = None, None
                for sub in child:
                    if sub.tag.endswith('key'): k = sub.text
                    elif sub.tag.endswith('value'): v = sub.text
                if k: props[k] = v
        return props

    def _determine_readable_type(self, elem, tag, props):
        atype = props.get('activityType', 'Unknown')
        if tag == 'startEvent': return "Start Timer" if elem.find('.//bpmn2:timerEventDefinition', NAMESPACES) is not None else "Start Message"
        if tag == 'endEvent': return "End"
        if tag == 'callActivity':
            mapping = {'Mapping': 'Message Mapping', 'Script': 'Groovy Script', 'Enricher': 'Content Modifier', 
                       'Splitter': 'Splitter', 'XmlModifier': 'XML Modifier', 'ProcessCallElement': 'Process Call', 
                       'Filter': 'Filter', 'DBStorage': 'Data Store','LocIntegration Process':'Local Integration Process'}
            return mapping.get(atype, atype)
        if tag == 'serviceTask':
            if atype == 'ExternalCall': return "Request Reply"
            if atype == 'contentEnricherWithLookup': return "Content Enricher"
            if atype == 'Send': return "Send"
        if 'Gateway' in tag: return "Router"
        return atype if atype != 'Unknown' else tag

    def get_resource_content(self, filename):
        if not filename: return None
        search_paths = [f"src/main/resources/{sub}/{filename}" for sub in ['script', 'groovy', 'mapping', 'xsd']]
        for path in search_paths:
            try:
                with self.zip_file.open(path) as f: return f.read().decode('utf-8', errors='ignore')
            except KeyError: continue
        return None

    def get_step_tech_data(self, node):
        props, ntype, nid = node['props'], node['type'], node['id']
        if ntype == 'Groovy Script':
            f = props.get('script')
            c = self.get_resource_content(f)
            return f"Script: {f}\nCode snippet:\n{c[:400]}" if c else "Script file content missing."
        elif ntype == 'Content Modifier':
            return f"Props: {props.get('propertyTable', '')}\nHeaders: {props.get('headerTable', '')}"
        elif ntype in ['Request Reply', 'Content Enricher', 'Send']:
            ap = self.adapter_configs.get(nid, {})
            full = {**props, **ap}
            return f"Adapter: {full.get('adapterId')}\nAddress: {full.get('address')}\nResource: {full.get('resourcePath')}"
        elif "Mapping" in ntype:
            return f"Mapping: {props.get('mappingname')}"
        return "Generic integration step."

    def find_start_node(self):
        for proc in self.root.findall('.//bpmn2:process', NAMESPACES):
            start = proc.find('bpmn2:startEvent', NAMESPACES)
            if start is not None: return start.get('id')
        return None

    def process(self):
        curr = self.find_start_node()
        if not curr:
            print("  [!] Could not find start node.")
            return

        queue = [curr]
        visited = set()
        functional_summaries = []
        
        print(f"  [+] Analyzing steps for {self.zip_path.name}...")
        
        while queue and self.step_counter < MAX_STEPS:
            curr = queue.pop(0)
            if curr in visited or curr not in self.elements: continue
            visited.add(curr)
            
            node = self.elements[curr]
            self.step_counter += 1
            
            tech_data = self.get_step_tech_data(node)
            summary = self.ai.get_step_summary(node['type'], tech_data)
            
            functional_summaries.append(f"Step {self.step_counter}: {node['name']} ({node['type']}) -> {summary}")
            self.rag_data["steps"].append({
                "step_index": self.step_counter,
                "type": node['type'],
                "description": summary
            })

            for link in node['outgoing']:
                if link['target'] not in visited:
                    queue.append(link['target'])

        print(f"  [+] Generating architectural summary...")
        full_context = "\n".join(functional_summaries)
        self.rag_data["architectural_analysis"] = self.ai.get_architectural_analysis(full_context)
        
        # Determine output filename: iflow_analysis_for_<basename>.json
        base_name = self.zip_path.stem
        output_name = f"iflow_analysis_for_{base_name}.json"
        
        with open(output_name, 'w', encoding='utf-8') as f:
            json.dump(self.rag_data, f, indent=4)
        print(f"  [OK] Saved to {output_name}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python cpi_batch_analyzer.py file1.zip file2.zip ...")
        return
    
    ai = DeepSeekAIHelper(DEEPSEEK_API_KEY)
    
    for zip_path in sys.argv[1:]:
        print(f"\n--- Processing: {zip_path} ---")
        analyzer = CPIFlowAnalyzer(zip_path, ai)
        if analyzer.load():
            analyzer.process()
        else:
            print(f"  [!] Skipping {zip_path}")

if __name__ == "__main__":
    main()
