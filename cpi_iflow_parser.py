#!/usr/bin/env python3
"""
SAP CPI AI Explainer - RAG Generator
1. PASS 1: Step-by-step AI analysis using a fast model (e.g., llama3.1).
2. PASS 2: Architectural grouping using a reasoning model (e.g., qwen3).
3. OUTPUT: Generates a structured JSON file for RAG training.
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
API_URL = "http://localhost:11434/api/chat"
STEP_MODEL = "llama3.1:8b" 
ARCHITECT_MODEL = "llama3.1:8b"
MAX_STEPS = 1000 
OUTPUT_FILE = "cpi_analysis_output.json"
# ---------------------

NAMESPACES = {
    'bpmn2': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
    'ifl': 'http:///com.sap.ifl.model/Ifl.xsd',
}
for prefix, uri in NAMESPACES.items():
    ET.register_namespace(prefix, uri)

# --- PROMPTS ---
ARCHITECT_SYSTEM_PROMPT = """You are an SAP CPI Integration Architect.

Your task is to generate a HIGH-QUALITY INTEGRATION DESCRIPTION based on the steps and
that will be used for semantic search and retrieval.

STRICT RULES:
- Output ONLY the description.
- Do NOT list steps or phases.
- Do NOT explain implementation details.
- Do NOT speculate or add business justification.
- Do NOT use generic phrases like “various processing” or “further handling”.

WHAT TO INCLUDE:
- Source system(s)
- Target system(s)
- Type of data exchanged
- Key technical behaviors (e.g., scheduled, event-driven, conditional, enrichment)
- Any notable constraints or patterns (e.g., routing, transformation, validation)

STYLE:
- 2–5 concise sentences.
- Technical, concrete, and factual.
- Optimized for semantic similarity, not human storytelling.

OUTPUT FORMAT:
- Plain text only. No headings. No bullet points."""

class AIHelper:
    def __init__(self):
        self.url = API_URL

    def get_step_summary(self, context, data):
        """Pass 1: Get concise explanation of a single step."""
        if not data: return "(No technical data)"
        
        prompt = f"Explain this '{context}' step. TECHNICAL DATA:\n{data}"
        payload = {
            "model": STEP_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": "Tell in 2-4 lines only:- Clearly state what the step does there.- Mention important fields, expressions, endpoints, or conditions if present."},
                {"role": "user", "content": prompt}
            ]
        }
        return self._call_ollama(payload)

    def get_architectural_analysis(self, full_flow_text):
        """Pass 2: Group the summarized flow into architectural phases."""
        prompt = (
            "Analyze the following SAP CPI iFlow.\n\n"
    "Task:\n"
    "Generate a concise, high-quality INTEGRATION DESCRIPTION based on the steps provided,\n"
    "that can be used for semantic search (RAG).\n\n"
    "Rules:\n"
    "- Output ONLY the description.\n"
    "- Do NOT list steps or phases.\n"
    "- Do NOT speculate about business intent.\n"
    "- Do NOT use generic phrases (e.g., 'various processing').\n"
    "- Be concrete and technical.\n\n"
    "Include:\n"
    "- Source system(s)\n"
    "- Target system(s)\n"
    "- Type of data exchanged\n"
    "- Key technical behaviors (scheduled, conditional, enrichment, routing, transformation)\n\n"
    "Style:\n"
    "- 2 to 4 sentences.\n"
    "- Plain text only.\n\n"
            f"iFlow steps:\n{full_flow_text}"
        )
        
        payload = {
            "model": ARCHITECT_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        }
        print(f"\n🧠 Sending aggregated context to Architect ({ARCHITECT_MODEL})...")
        return self._call_ollama(payload)

    def _call_ollama(self, payload):
        try:
            req = urllib.request.Request(
                self.url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result["message"]["content"].strip()
        except Exception as e:
            return f"(AI Error: {str(e)})"

class CPIFlowAnalyzer:
    def __init__(self, zip_path):
        self.zip_path = Path(zip_path)
        self.xml_content = None
        self.root = None
        self.elements = {}
        self.adapter_configs = {} 
        self.parameters = {}       
        self.step_counter = 0
        self.zip_file = None
        self.ai = AIHelper()
        self.rag_data = {
            "iflow_file": self.zip_path.name,
            "steps": [],
            "architectural_analysis": ""
        }

    def load(self):
        try:
            self.zip_file = zipfile.ZipFile(self.zip_path, 'r')
            self._load_parameters()
            candidates = [f for f in self.zip_file.namelist() if f.endswith('.iflw') and 'integrationflow' in f.lower()]
            if not candidates: raise Exception("No .iflw found")
            with self.zip_file.open(candidates[0]) as f:
                self.xml_content = f.read()
            self.root = ET.fromstring(self.xml_content)
            self._parse_elements()
        except Exception as e:
            print(f"Error loading ZIP: {e}")
            sys.exit(1)

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
        for mf in self.root.findall('.//bpmn2:messageFlow', NAMESPACES):
            props = self._get_ifl_properties(mf)
            props = self._resolve_placeholders(props)
            s, t = mf.get('sourceRef'), mf.get('targetRef')
            if s: self.adapter_configs[s] = props
            if t: self.adapter_configs[t] = props

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
                'element': elem,
                'props': props,
                'outgoing': []
            }
            
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
            if atype == 'Mapping': return "Message Mapping"
            if atype == 'Script': return "Groovy Script"
            if atype == 'Enricher': return "Content Modifier"
            if atype == 'Splitter': return "Splitter"
            if atype == 'XmlModifier': return "XML Modifier"
            if atype == 'ProcessCallElement': return "Process Call"
            if atype == 'Filter': return "Filter"
            if atype == 'DBStorage': return "Data Store"
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
        name = node['name']
        props = node['props']
        ntype = node['type']
        nid = node['id']
        
        # Build concise tech string for Step AI
        if ntype == 'Groovy Script':
            f = props.get('script')
            c = self.get_resource_content(f)
            return f"Script: {f}\nCode:\n{c[:500]}..." if c else "File not found"
        elif ntype == 'Content Modifier':
            p = props.get('propertyTable') or ''
            h = props.get('headerTable') or ''
            return f"Props: {p}\nHeaders: {h}"
        elif ntype in ['Request Reply', 'Content Enricher', 'Send']:
            ap = self.adapter_configs.get(nid, {})
            full = {**props, **ap}
            return f"System: {full.get('address')}\nEntity: {full.get('resourcePath')}\nQuery: {full.get('queryOptions')}"
        elif "Mapping" in ntype:
            return f"Mapping File: {props.get('mappingname')}"
        elif ntype == 'Router':
            return "Routing Decision"
        return "Standard Step"

    def find_start_node(self):
        for proc in self.root.findall('.//bpmn2:process', NAMESPACES):
            if 'Integration Process' in proc.get('name', ''):
                start = proc.find('bpmn2:startEvent', NAMESPACES)
                if start is not None: return start.get('id')
        return None

    def run_two_pass_analysis(self):
        curr = self.find_start_node()
        if not curr: return

        print(f"🚀 PASS 1: Analyzing steps one-by-one ({STEP_MODEL})...")
        
        # Using a stack for Depth-First Traversal (to cover all branches)
        # Each item is (node_id)
        # We also need a way to order them roughly sequentially for the report, 
        # but DFS naturally follows flow.
        
        # NOTE: For RAG purposes, a simple linear walk of the main path + side paths is okay.
        # But a true graph walk is safer to not miss detached ends.
        # Let's stick to the queue/stack based traversal we discussed.
        
        queue = [curr]
        visited = set()
        functional_summaries = []
        
        while queue and self.step_counter < MAX_STEPS:
            curr = queue.pop(0) # BFS style (Process level by level-ish)
            
            if curr in visited: continue
            visited.add(curr)
            
            node = self.elements[curr]
            self.step_counter += 1
            print(f"   Step {self.step_counter}: {node['name']} ({node['type']})", end="...", flush=True)
            
            # 1. Get Tech Data
            tech_data = self.get_step_tech_data(node)
            
            # 2. Ask Step AI
            summary = self.ai.get_step_summary(node['type'], tech_data)
            print(f" -> {summary}")
            
            # 3. Store for Pass 2 AND RAG Data
            summary_text = f"Step {self.step_counter}: {node['name']} ({node['type']}) -> {summary}"
            functional_summaries.append(summary_text)
            
            # Add to Structured Data (SIMPLIFIED AS REQUESTED)
            self.rag_data["steps"].append({
                "step_number": self.step_counter,
                "type": node['type'],
                "ai_summary": summary
            })

            # Move next (Find all children)
            if node['outgoing']:
                for link in node['outgoing']:
                    target_id = link['target']
                    if target_id not in visited:
                        queue.append(target_id)

        print(f"\n🚀 PASS 2: Sending aggregated flow to Architect ({ARCHITECT_MODEL})...")
        full_context = "\n".join(functional_summaries)
        
        final_analysis = self.ai.get_architectural_analysis(full_context)
        
        # Store analysis
        self.rag_data["architectural_analysis"] = final_analysis
        
        print("\n" + "="*60)
        print("🏛️  ARCHITECTURAL ANALYSIS REPORT")
        print("="*60)
        print(final_analysis)
        print("="*60)
        
        # SAVE JSON
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.rag_data, f, indent=4)
        print(f"\n✅ RAG Data saved to: {OUTPUT_FILE}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python cpi_ai_explainer.py <file.zip>")
        return
    
    analyzer = CPIFlowAnalyzer(sys.argv[1])
    analyzer.load()
    analyzer.run_two_pass_analysis()

if __name__ == "__main__":
    main()
