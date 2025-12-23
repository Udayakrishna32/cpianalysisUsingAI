#!/usr/bin/env python3
"""
SAP CPI Structured Analyzer (Reasoning-Ready v1.2)
Combines:
1. Robust Stack-based Recursive Traversal
2. Reasoning-Ready JSON Prompts
3. Deterministic Graph & Contract Enforcement (Anti-Hallucination)
4. Fix: Allow re-traversal of shared sub-processes (Graph vs Tree logic)
5. Fix: Correct Execution Order for Process Calls (Trace Subprocess BEFORE Next Step)
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

# ================= CONFIGURATION =================
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

# ================= DETERMINISTIC CONTRACTS =================

# 1. Hard-coded Roles (The LLM should not guess these)
STEP_ROLES = {
    "Start Timer": "control",
    "Start Message": "control", # Trigger
    "End": "control",
    "Router": "routing",
    "Sequential Multicast": "routing",
    "Filter": "routing", # Or validation
    "Join": "routing",
    "Process Call": "orchestration",
    "Looping Process": "orchestration",
    "Request Reply": "orchestration", # External call
    "Send": "delivery",
    "Content Modifier": "transformation",
    "Groovy Script": "transformation",
    "Message Mapping": "transformation",
    "Splitter": "transformation",
    "Encoder": "transformation",
    "Decoder": "transformation",
    "PGP Encrypt": "security",
    "PGP Decrypt": "security",
    "Signer": "security",
    "Data Store": "persistence",
    "Write Variables": "persistence",
    "Read Variables": "persistence"
}

# ================= PROMPTS =================

STEP_SYSTEM_PROMPT = """
You are an SAP CPI static analyzer.

Convert ONE CPI step into a STRICT JSON NODE
using EXACTLY this schema:

{
  "id": "",
  "type": "",
  "role": "",
  "reads": {},
  "writes": {},
  "control": {},
  "error_handling": {},
  "metadata": {}
}

RULES:
- Output VALID JSON ONLY
- Do NOT populate 'outgoing' or 'edges' in control/metadata (this is handled deterministically).
- Focus on extracting LOGIC (e.g., script names, mapping names, precise conditions).
- For 'reads' and 'writes', infer specific variable names if visible in Technical Data.
"""

ARCHITECT_SYSTEM_PROMPT = """
You are an SAP CPI integration architect.

Given a list of structured nodes and a graph of edges, infer HIGH-LEVEL INTENT.

Output STRICT JSON ONLY:

{
  "integration_pattern": [],
  "trigger_type": "",
  "source_systems": [],
  "target_systems": [],
  "cross_cutting_concerns": [],
  "state_management": "",
  "delivery_mode": ""
}

RULES:
- Infer intent only
- Do NOT describe steps
"""

# ================= AI HELPER =================

class DeepSeekAIHelper:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = API_URL

    def _call_api(self, messages):
        payload = {
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "response_format": {"type": "json_object"} 
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
                print(f"API HTTP Error: {e.code}")
                raise
            except Exception as e:
                print(f"API General Error: {str(e)}")
                raise
        raise RuntimeError("Max retries exceeded")

    def get_step_node(self, node_info, tech_data):
        user_content = json.dumps({
            "step_id": node_info['id'],
            "step_name": node_info['name'],
            "step_type": node_info['type'],
            "process": node_info['process_name'],
            "technical_data": tech_data
            # NOTE: We do NOT send 'outgoing' to LLM to prevent it from hallucinating control flow.
            # We handle edges deterministically in the Analyzer.
        }, indent=2)

        messages = [
            {"role": "system", "content": STEP_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        return self._call_api(messages)

    def get_architecture(self, nodes_list, edges_list):
        # We pass both nodes and edges to the architect
        context = {
            "nodes": nodes_list,
            "edges": edges_list
        }
        messages = [
            {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, indent=2)}
        ]
        return self._call_api(messages)

# ================= ANALYZER =================

class CPIFlowAnalyzer:
    def __init__(self, zip_path, ai_helper):
        self.zip_path = Path(zip_path)
        self.ai = ai_helper
        self.xml_content = None
        self.root = None
        self.elements = {}
        self.processes = {} 
        self.adapter_configs = {} 
        self.parameters = {}        
        self.step_counter = 0
        self.zip_file = None
        self.main_process_id = None
        
        # CHANGED: 'analyzed_nodes' tracks which nodes have JSON definitions generated.
        # We no longer use a global 'visited' for traversal blocking, to allow multiple paths to touch the same node.
        self.analyzed_nodes = set() 
        self.traversal_stack = set() # Prevents cycles (infinite recursion) in current path
        
        # New Output Structure
        self.output = {
            "iflow": self.zip_path.name,
            "nodes": [],
            "edges": [], # Graph topology
            "architecture": {}
        }

    # ---------- LOADING & PARSING ----------
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
            
            starts = process.findall('.//bpmn2:startEvent', NAMESPACES)
            best_start = None
            for s in starts:
                if s.find('.//bpmn2:errorEventDefinition', NAMESPACES) is not None:
                    continue
                best_start = s.get('id')
                if s.find('.//bpmn2:timerEventDefinition', NAMESPACES) is not None:
                    break 
            
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
                    'process_type': ptype, # Important for End Event logic
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
            header = props.get('headerTable', '')
            return f"Properties: {table} Headers: {header}"
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

    # ---------- ENRICHMENT LOGIC (DETERMINISTIC) ----------

    def _enrich_node(self, node_json, element, tech_data):
        """Enforces schema contracts and fixes roles/data based on type."""
        
        # 1. Deterministic Role (Fix Semantic Errors)
        if element['type'] in STEP_ROLES:
            node_json['role'] = STEP_ROLES[element['type']]
        
        # 2. Mandatory Reads/Writes Contracts
        # Initialize if missing
        if 'reads' not in node_json: node_json['reads'] = {}
        if 'writes' not in node_json: node_json['writes'] = {}

        if element['type'] == 'Content Modifier':
            # Reads payload/headers, Writes payload/properties
            node_json['reads'].setdefault('payload', True)
            node_json['reads'].setdefault('headers', True)
            node_json['writes'].setdefault('payload', True)
            node_json['writes'].setdefault('properties', True)
        
        elif element['type'] in ['Groovy Script', 'Message Mapping']:
            node_json['reads'].setdefault('payload', True)
            node_json['writes'].setdefault('payload', True)
        
        elif element['type'] in ['Request Reply', 'Send']:
            node_json['reads'].setdefault('payload', True)
            node_json['writes'].setdefault('payload', True) # Response overwrites payload usually
        
        elif element['type'] == 'Router':
            node_json['reads'].setdefault('payload', True) # Often reads XML for routing
        
        elif element['type'] == 'Data Store':
            op = element['props'].get('operation', '').lower()
            if 'put' in op:
                node_json['reads'].setdefault('payload', True)
                node_json['writes'].setdefault('datastore', True)
            elif 'select' in op or 'get' in op:
                node_json['reads'].setdefault('datastore', True)
                node_json['writes'].setdefault('payload', True)

        # 3. Semantic End Types
        if element['type'] == 'End':
            if 'metadata' not in node_json: node_json['metadata'] = {}
            if element.get('process_type') == 'IntegrationProcess': # Top level
                node_json['metadata']['end_type'] = 'flow_termination'
            else: # Local Process
                node_json['metadata']['end_type'] = 'subprocess_return'

        # 4. Control Flow Injection (Moving outgoing to Control)
        # Note: We construct 'branch_conditions' in the trace loop, but we can init here
        if 'control' not in node_json: node_json['control'] = {}
        
        return node_json

    # ---------- RECURSIVE TRAVERSAL ----------

    def trace(self, node_id):
        """Recursive DFS tracing with Graph Generation"""
        if self.step_counter >= MAX_STEPS:
            return

        # CYCLIC CHECK: Only stop if we are in a loop within the CURRENT recursion stack
        if node_id in self.traversal_stack:
            return
        
        node = self.elements.get(node_id)
        if not node: return

        self.step_counter += 1
        
        # Add to stack for this path
        self.traversal_stack.add(node_id)

        # --- NODE GENERATION (Only if not already analyzed) ---
        # We process the node to generate edges every time, but we only ask AI for JSON once.
        
        structured_node = None
        
        if node_id not in self.analyzed_nodes:
            self.analyzed_nodes.add(node_id)
            tech_data = self.get_tech_data(node)
            print(f"    [{self.step_counter}] {node['process_name']} > {node['type']}: {node['name'][:30]}")

            # --- AI CALL ---
            try:
                raw_json = self.ai.get_step_node(node, tech_data)
                structured_node = json.loads(raw_json)
            except Exception as e:
                print(f"      [!] AI Parse Error for {node_id}: {e}")
                structured_node = {
                    "id": node["id"],
                    "type": node["type"],
                    "role": "unknown",
                    "metadata": {"error": "AI_PARSE_FAIL"}
                }

            # --- DETERMINISTIC ENRICHMENT ---
            structured_node = self._enrich_node(structured_node, node, tech_data)
            
            # Add to Nodes List
            self.output["nodes"].append({
                "order": self.step_counter, 
                "process": node["process_name"],
                "node": structured_node
            })
        else:
             # Node already exists, but we still need to calculate control flow for edges
             pass

        # --- GRAPH & CONTROL FLOW CONSTRUCTION ---
        outgoing_control = []

        # 1. SPECIAL PRIORITY: Process Call -> Trace Subprocess FIRST (Depth-First)
        # This matches execution semantics: Enter Subprocess -> Return -> Continue
        if node['type'] in ['Process Call', 'Looping Process']:
            sub_pid = node['props'].get('processId')
            sub_start = self.processes.get(sub_pid, {}).get('start')
            if sub_start:
                # Add Implicit Edge for Graph
                edge_entry = {
                    "from": node["id"],
                    "to": sub_start,
                    "type": "ProcessCall"
                }
                if edge_entry not in self.output["edges"]:
                    self.output["edges"].append(edge_entry)
                
                # Recurse DOWN into subprocess
                self.trace(sub_start)
        
        # 2. HANDLE OUTGOING (Continuation / Branching)
        # 2a. Sequential Multicast (Ordered Branches)
        if node['type'] == 'Sequential Multicast':
            table = node['props'].get('routingSequenceTable', '')
            matches = re.findall(r'<cell>(\d+)</cell><cell>(SequenceFlow_\d+)</cell>', table)
            sorted_flows = sorted(matches, key=lambda x: int(x[0]))
            
            for _, sf_id in sorted_flows:
                target_id = next((l['target'] for l in node['outgoing'] if l['sf_id'] == sf_id), None)
                if target_id:
                    # Graph Edge
                    edge_entry = {
                        "from": node["id"],
                        "to": target_id,
                        "type": "SequenceFlow",
                        "metadata": {"order": int(_)}
                    }
                    if edge_entry not in self.output["edges"]:
                         self.output["edges"].append(edge_entry)
                    
                    # Control Block
                    outgoing_control.append({"target": target_id, "type": "sequence", "order": int(_)})
                    self.trace(target_id)
        
        # 2b. Router (Conditional Branches)
        elif node['type'] == 'Router':
            for link in node['outgoing']:
                # Graph Edge
                edge_entry = {
                    "from": node["id"],
                    "to": link['target'],
                    "type": "SequenceFlow",
                    "condition": link['condition']
                }
                if edge_entry not in self.output["edges"]:
                        self.output["edges"].append(edge_entry)

                # Control Block
                outgoing_control.append({
                    "target": link['target'], 
                    "condition": link['condition']
                })
                
                self.trace(link['target'])
        
        # 2c. Standard Flow
        else:
            for link in node['outgoing']:
                # Graph Edge
                edge_entry = {
                    "from": node["id"],
                    "to": link['target'],
                    "type": "SequenceFlow"
                }
                if edge_entry not in self.output["edges"]:
                        self.output["edges"].append(edge_entry)

                # Control Block
                outgoing_control.append({"target": link['target']})
                
                self.trace(link['target'])

        # Inject constructed control flow into JSON (Only if we just created the node)
        if structured_node and outgoing_control:
            structured_node['control']['next'] = outgoing_control

        # Remove from stack as we backtrack
        self.traversal_stack.remove(node_id)

    # ---------- EXECUTION ----------

    def run(self):
        start_node = self.processes.get(self.main_process_id, {}).get('start')
        if not start_node:
            print("  [!] Could not find trigger in Participant_Process_1")
            return

        print(f"  [+] Tracing execution path for {self.zip_path.name}...")
        self.trace(start_node)

        print(f"  [+] Generating High-Level Architecture...")
        try:
            arch_raw = self.ai.get_architecture(self.output["nodes"], self.output["edges"])
            self.output["architecture"] = json.loads(arch_raw)
        except Exception as e:
            print(f"  [!] Architecture Parse Error: {e}")
            self.output["architecture"] = {"error": "Failed to parse architecture JSON"}

        out = f"structured_{self.zip_path.stem}.json"
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(self.output, f, indent=2)
        print(f"  [OK] Generated structured output: {out}")

# ================= MAIN =================

def main():
    if len(sys.argv) < 2:
        print("Usage: python cpi_structured_analyzer.py <iflow.zip>")
        return

    ai = DeepSeekAIHelper(DEEPSEEK_API_KEY)

    for zip_path in sys.argv[1:]:
        print(f"\n--- Analyzing: {zip_path} ---")
        analyzer = CPIFlowAnalyzer(zip_path, ai)
        if analyzer.load():
            analyzer.run()

if __name__ == "__main__":
    main()
