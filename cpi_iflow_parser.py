#!/usr/bin/env python3
"""
SAP CPI AI Explainer - Two-Pass Architecture
1. PASS 1: Step-by-step AI analysis (Technical -> Functional Summary).
2. PASS 2: Architectural grouping (Functional Summaries -> Phases).
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

# MODEL 1: Fast model for individual steps
STEP_MODEL = "llama3.1:8b" 

# MODEL 2: Reasoning model for final grouping (as requested)
ARCHITECT_MODEL = "llama3.1:8b"

MAX_STEPS = 1000 
# ---------------------

NAMESPACES = {
    'bpmn2': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
    'ifl': 'http:///com.sap.ifl.model/Ifl.xsd',
}
for prefix, uri in NAMESPACES.items():
    ET.register_namespace(prefix, uri)

# --- PROMPTS ---
ARCHITECT_SYSTEM_PROMPT = """Analyze the following SAP CPI iFlow.

Requirements:
1. Group the steps into logical ARCHITECTURAL PHASES.
2. Give each phase a clear purpose (why it exists) in 2-5 lines.
3. Explain how data flows between phases in 2-5 lines.
4. Call out which phase contains CORE BUSINESS LOGIC in 2-5 lines.
5. Identify important validations and routing decisions in 2-5 lines.
6. Mention any redundant or low-value steps in 2-5 lines.
7. Provide a short “mental model” of the overall flow in 2-5 lines (important).

Do NOT explain every step individually.
Do NOT repeat CPI documentation.
Focus on understanding, not description."""

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
                {"role": "system", "content": "You are a concise SAP CPI expert. Summarize the logic in ONE to Three simple sentence. Do not miss fields or logics used in those. Just say what it does."},
                {"role": "user", "content": prompt}
            ]
        }
        return self._call_ollama(payload)

    def get_architectural_analysis(self, full_flow_text):
        """Pass 2: Group the summarized flow into architectural phases."""
        prompt = (
            "Analyze the following SAP CPI iFlow.\n\n"
            "Requirements:\n"
            "1. Group the steps into logical ARCHITECTURAL PHASES .\n"
            "2. Give each phase a clear purpose (why it exists) in 2-5 lines.\n"
            "3. Explain how data flows between phases in 2-5 lines.\n"
            "4. Call out which phase contains CORE BUSINESS LOGIC in 2-5 lines.\n"
            "5. Identify important validations and routing decisions in 2-5 lines.\n"
            "6. Mention any redundant or low-value steps in 2-5 lines.\n"
            "7. Provide a short “mental model” of the overall flow.\n\n"
            "Do NOT explain every step individually.\n"
            "Do NOT repeat CPI documentation.\n"
            "Focus on understanding, not description.\n\n"
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
        
        visited = set()
        functional_summaries = []
        
        while curr and self.step_counter < MAX_STEPS:
            if curr in visited: break
            visited.add(curr)
            
            node = self.elements[curr]
            self.step_counter += 1
            print(f"   Step {self.step_counter}: {node['name']} ({node['type']})", end="...", flush=True)
            
            # 1. Get Tech Data
            tech_data = self.get_step_tech_data(node)
            
            # 2. Ask Step AI
            summary = self.ai.get_step_summary(node['type'], tech_data)
            print(f" -> {summary}")
            
            # 3. Store for Pass 2
            functional_summaries.append(f"Step {self.step_counter}: {node['name']} ({node['type']}) -> {summary}")

            # Move next
            if node['outgoing']:
                links = node['outgoing']
                nxt = links[0]['target']
                if len(links) > 1:
                    for l in links:
                        t = self.elements.get(l['target'])
                        if t and t['type'] != 'End':
                            nxt = l['target']
                            break
                curr = nxt
            else:
                curr = None

        print(f"\n🚀 PASS 2: Sending aggregated flow to Architect ({ARCHITECT_MODEL})...")
        full_context = "\n".join(functional_summaries)
        
        final_analysis = self.ai.get_architectural_analysis(full_context)
        
        print("\n" + "="*60)
        print("🏛️  ARCHITECTURAL ANALYSIS REPORT")
        print("="*60)
        print(final_analysis)
        print("="*60)

def main():
    if len(sys.argv) < 2:
        print("Usage: python cpi_ai_explainer.py <file.zip>")
        return
    
    analyzer = CPIFlowAnalyzer(sys.argv[1])
    analyzer.load()
    analyzer.run_two_pass_analysis()

if __name__ == "__main__":
    main()
