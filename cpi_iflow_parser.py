#!/usr/bin/env python3
"""
SAP CPI AI Explainer - Batch Processor
Implementation: Stack-based Recursive Traversal
1. Starts at Participant_Process_1, explicitly avoiding Exception SubProcesses for the entry point.
2. Follows ordered branches in Sequential Multicasts.
3. Jumps into Local Processes via Process Calls and returns upon 'End' event.
4. Uses DeepSeek API for step-by-step and architectural analysis.
"""

import zipfile
import xml.etree.ElementTree as ET
import sys
import json
import time
import urllib.request
import urllib.error
import re
from pathlib import Path

# --- CONFIGURATION ---
DEEPSEEK_API_KEY = "sk-f7d101c4a97246318ab270f5d67abfdd"
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
MAX_STEPS = 1500 

NAMESPACES = {
    'bpmn2': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
    'ifl': 'http:///com.sap.ifl.model/Ifl.xsd',
}
for prefix, uri in NAMESPACES.items():
    ET.register_namespace(prefix, uri)

# --- PROMPTS ---
ARCHITECT_SYSTEM_PROMPT = """You are an SAP CPI Integration Architect.
Your task is to generate a HIGH-QUALITY INTEGRATION DESCRIPTION based on the steps provided.
This will be used for semantic search and retrieval (RAG).

STRICT RULES:
- Output ONLY the description.
- Be concrete, technical, and factual.
- Describe the end-to-end data flow from source to target.

STYLE:
- 2–4 concise sentences.
- Plain text only. No bullet points."""

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
                if e.code == 429:
                    wait = 2 ** i
                    time.sleep(wait)
                    continue
                return f"(API Error: {e.code})"
            except Exception as e:
                return f"(Error: {str(e)})"
        return "(Error: Max retries exceeded)"

    def get_step_summary(self, context, data, process_name):
        system_msg = f"You are an SAP CPI expert. Briefly explain what this step does within the context of the '{process_name}' process.produce ONE single-paragraph, implementation-accurate description of the CPI step., example (style reference only): Reads exchange property P_LastSuccessfulRunDate, writes exchange property P_QueryFilter using the resolved timestamp, does not modify the message body or headers, persists no state beyond runtime, requires P_LastSuccessfulRunDate to be present in the exchange context, and throws a runtime exception if the property is missing or invalid etc etc like this"
        user_msg = f"Step Type: {context}\nTechnical Data:\n{data}"
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
        return self._call_api(messages)

    def get_architectural_analysis(self, full_flow_text):
        messages = [
            {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Summarize this iFlow logic:\n\n{full_flow_text}"}
        ]
        return self._call_api(messages)

class CPIFlowAnalyzer:
    def __init__(self, zip_path, ai_helper):
        self.zip_path = Path(zip_path)
        self.ai = ai_helper
        self.xml_content = None
        self.root = None
        self.elements = {}
        self.processes = {} # pid -> {start, name, type}
        self.adapter_configs = {} 
        self.parameters = {}       
        self.step_counter = 0
        self.zip_file = None
        self.main_process_id = None
        self.visited_nodes = set()
        self.functional_summaries = []
        self.rag_data = {
            "iflow_file": self.zip_path.name,
            "steps": [],
            "architectural_analysis": ""
        }

    def load(self):
        try:
            self.zip_file = zipfile.ZipFile(self.zip_path, 'r')
            self._load_parameters()
            candidates = [f for f in self.zip_file.namelist() if f.endswith('.iflw')]
            if not candidates: return False
            iflow_file = next((c for c in candidates if 'integrationflow' in c.lower()), candidates[0])
            with self.zip_file.open(iflow_file) as f:
                self.xml_content = f.read()
            self.root = ET.fromstring(self.xml_content)
            self._parse_elements()
            return True
        except Exception as e:
            print(f"  [!] Load Error: {e}")
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

    def _resolve(self, val):
        if isinstance(val, str) and '{{' in val:
            clean = val.replace('{{', '').replace('}}', '').strip()
            return self.parameters.get(clean, val)
        return val

    def _parse_elements(self):
        # 1. Map Participants
        process_map = {}
        for part in self.root.findall('.//bpmn2:collaboration/bpmn2:participant', NAMESPACES):
            ref = part.get('processRef')
            ptype = part.get('ifl:type')
            if part.get('id') == "Participant_Process_1":
                self.main_process_id = ref
            if ref: process_map[ref] = (ptype, part.get('name'))

        # 2. Map Message Flows (Adapters)
        for mf in self.root.findall('.//bpmn2:messageFlow', NAMESPACES):
            props = {k: self._resolve(v) for k, v in self._get_props(mf).items()}
            s, t = mf.get('sourceRef'), mf.get('targetRef')
            if s: self.adapter_configs[s] = props
            if t: self.adapter_configs[t] = props

        # 3. Parse Processes
        for process in self.root.findall('.//bpmn2:process', NAMESPACES):
            pid = process.get('id')
            ptype, pname = process_map.get(pid, ("Local", process.get('name')))
            
            # Identify the CORRECT start node (Avoid Error Starts for the main entry)
            starts = process.findall('.//bpmn2:startEvent', NAMESPACES)
            best_start = None
            for s in starts:
                # Prioritize Timer or standard starts
                if s.find('.//bpmn2:errorEventDefinition', NAMESPACES) is not None:
                    continue
                best_start = s.get('id')
                if s.find('.//bpmn2:timerEventDefinition', NAMESPACES) is not None:
                    break # Timer is usually the main entry if present
            
            self.processes[pid] = {
                'start': best_start,
                'name': pname,
                'type': ptype
            }

            for elem in process.findall('.//*[@id]'):
                eid = elem.get('id')
                tag = elem.tag.split('}')[-1]
                if tag in ['sequenceFlow', 'incoming', 'outgoing']: continue
                
                props = {k: self._resolve(v) for k, v in self._get_props(elem).items()}
                self.elements[eid] = {
                    'id': eid,
                    'name': elem.get('name', '') or eid,
                    'tag': tag,
                    'process_id': pid,
                    'process_name': pname,
                    'type': self._determine_type(elem, tag, props),
                    'props': props,
                    'outgoing': []
                }
            
            for sf in process.findall('.//bpmn2:sequenceFlow', NAMESPACES):
                s, t = sf.get('sourceRef'), sf.get('targetRef')
                if s in self.elements and t in self.elements:
                    cond = sf.find('.//bpmn2:conditionExpression', NAMESPACES)
                    self.elements[s]['outgoing'].append({
                        'target': t, 
                        'condition': cond.text if cond is not None else "",
                        'sf_id': sf.get('id')
                    })

    def _get_props(self, elem):
        props = {}
        for prop in elem.iter():
            if prop.tag.endswith('property'):
                k, v = None, None
                for sub in prop:
                    if sub.tag.endswith('key'): k = sub.text
                    elif sub.tag.endswith('value'): v = sub.text
                if k: props[k] = v
        return props

    def _determine_type(self, elem, tag, props):
        atype = props.get('activityType', 'Unknown')
        if tag == 'startEvent':
            if elem.find('.//bpmn2:errorEventDefinition', NAMESPACES) is not None: return "Start Error"
            return "Start Timer" if elem.find('.//bpmn2:timerEventDefinition', NAMESPACES) is not None else "Start Message"
        if tag == 'endEvent': return "End"
        if tag == 'callActivity':
            mapping = {'Mapping': 'Message Mapping', 'Script': 'Groovy Script', 'Enricher': 'Content Modifier', 
                       'Splitter': 'Splitter', 'ProcessCallElement': 'Process Call', 'PgpEncrypt': 'PGP Encrypt',
                       'DBstorage': 'Data Store', 'LoopingProcess': 'Looping Process'}
            return mapping.get(atype, atype)
        if tag == 'serviceTask':
            return "Request Reply" if atype == 'ExternalCall' else "Send"
        if 'Gateway' in tag:
            return "Sequential Multicast" if atype == 'SequentialMulticast' else "Router"
        return atype

    def get_tech_data(self, node):
        props, ntype, nid = node['props'], node['type'], node['id']
        if ntype == 'Groovy Script':
            return f"Script File: {props.get('script', 'Unknown')}"
        elif ntype == 'Content Modifier':
            table = props.get('propertyTable', '')
            wrap = props.get('wrapContent', '') or ""
            return f"Logic: {table} {wrap[:200]}"
        elif ntype in ['Request Reply', 'Send']:
            ap = self.adapter_configs.get(nid, {})
            return f"Target: {ap.get('address', 'Unknown')} Path: {ap.get('resourcePath', '') or ap.get('queryOptions', '')}"
        elif ntype in ['Process Call', 'Looping Process']:
            return f"Target Process ID: {props.get('processId', 'Unknown')}"
        elif ntype == 'Sequential Multicast':
            return "Ordered Multicast Branching"
        elif ntype == 'Data Store':
            return f"Operation: {props.get('operation', 'Unknown')} Storage: {props.get('storageName', 'Unknown')}"
        elif ntype == 'Router':
            conds = [f"To {l['target']}: {l['condition']}" for l in node['outgoing'] if l['condition']]
            return "Branch conditions: " + ("; ".join(conds) if conds else "Default route.")
        return f"Standard {ntype} step."

    def trace(self, node_id):
        """Recursive DFS tracing with Call Stack logic."""
        if self.step_counter >= MAX_STEPS or node_id in self.visited_nodes:
            return

        node = self.elements.get(node_id)
        if not node: return

        # Mark node as visited only if it's NOT a start node (allows re-entry to local processes)
        if "Start" not in node['type']:
            self.visited_nodes.add(node_id)

        self.step_counter += 1
        tech_data = self.get_tech_data(node)
        summary = self.ai.get_step_summary(node['type'], tech_data, node['process_name'])
        
        print(f"    [{self.step_counter}] {node['process_name']} > {node['type']}: {node['name'][:30]}")

        self.rag_data["steps"].append({
            "step": self.step_counter,
            "type": node['type'],
            "description": summary
        })
        self.functional_summaries.append(f"{node['name']} ({node['type']}): {summary}")

        # 1. HANDLE PROCESS JUMPS
        if node['type'] in ['Process Call', 'Looping Process']:
            sub_pid = node['props'].get('processId')
            sub_start = self.processes.get(sub_pid, {}).get('start')
            if sub_start:
                self.trace(sub_start)

        # 2. HANDLE BRANCHING
        if node['type'] == 'Sequential Multicast':
            table = node['props'].get('routingSequenceTable', '')
            matches = re.findall(r'<cell>(\d+)</cell><cell>(SequenceFlow_\d+)</cell>', table)
            sorted_flows = sorted(matches, key=lambda x: int(x[0]))
            for _, sf_id in sorted_flows:
                target = next((l['target'] for l in node['outgoing'] if l['sf_id'] == sf_id), None)
                if target: self.trace(target)
        else:
            for link in node['outgoing']:
                if link['target'] not in self.visited_nodes:
                    self.trace(link['target'])

    def process(self):
        start_node = self.processes.get(self.main_process_id, {}).get('start')
        if not start_node:
            print("  [!] Could not find trigger in Participant_Process_1")
            return

        print(f"  [+] Tracing execution path for {self.zip_path.name}...")
        self.trace(start_node)

        print(f"  [+] Performing architectural analysis...")
        ctx = "\n".join(self.functional_summaries)
        self.rag_data["architectural_analysis"] = self.ai.get_architectural_analysis(ctx)
        
        out = f"iflow_analysis_for_{self.zip_path.stem}.json"
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(self.rag_data, f, indent=4)
        print(f"  [OK] Successfully analyzed {self.step_counter} steps. Output: {out}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python cpi_batch_analyzer.py <file.zip>")
        return
    ai = DeepSeekAIHelper(DEEPSEEK_API_KEY)
    for zip_path in sys.argv[1:]:
        print(f"\n--- Batch Item: {zip_path} ---")
        analyzer = CPIFlowAnalyzer(zip_path, ai)
        if analyzer.load():
            analyzer.process()

if __name__ == "__main__":
    main()
