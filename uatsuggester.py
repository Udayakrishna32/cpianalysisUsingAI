#!/usr/bin/env python3
"""
SAP CPI Variable Lifecycle Analyzer
Implementation: Stack-based Recursive Traversal (Preserved) with Variable Extraction Prompts
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
from datetime import datetime

# --- CONFIGURATION ---
DEEPSEEK_API_KEY = "sk-f7d101c4a97246318ab270f5d67abfdd" # Replace if needed or load from env
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

# Prompt for Variable Extraction (JSON)
VARIABLE_SYSTEM_PROMPT = """You are an SAP CPI Code Analyzer.
Your task is to analyze a specific Integration Step and extract variable usage.

INPUT: Technical data of a CPI step (XML attributes, Script names, Mapping configs).

OUTPUT: A strict JSON array containing objects for EVERY variable (Exchange Property, Header, Data Store) interacted with in this step.
If no variables are used, return an empty JSON array [].

JSON SCHEMA per variable:
{
    "name": "Variable Name (e.g., encryptionStatus, P_LastRunDate)",
    "scope": "One of ['Exchange Property', 'Header', 'Data Store', 'Global Variable']",
    "action": "One of ['CREATED', 'READ', 'MODIFIED', 'DELETED']",
    "purpose": "One of ['DECISION', 'CONFIGURATION', 'TRANSFORMATION', 'PERSISTENCE_QUERY', 'PERSISTENCE_WRITE', 'INITIALIZATION']",
    "context": "Brief explanation of how it is used in this specific step (max 10 words)"
}

RULES:
1. Identify properties from Content Modifiers (header/property tables).
2. Identify variables from Routers (in ${property.name}).
3. Identify Data Store names as variables (Scope: Data Store).
4. Do not explain the step textually. ONLY return JSON.
"""

# Prompt for Step Description (Text)
DESCRIPTION_SYSTEM_PROMPT = """You are an SAP CPI expert. Briefly explain what this step does within the context of the '{process_name}' process.produce ONE single-paragraph, implementation-accurate description of the CPI step., example (style reference only): Reads exchange property P_LastSuccessfulRunDate, writes exchange property P_QueryFilter using the resolved timestamp, does not modify the message body or headers, persists no state beyond runtime, requires P_LastSuccessfulRunDate to be present in the exchange context, and throws a runtime exception if the property is missing or invalid etc etc like this"""

# Prompt for UAT Generation (Text)
UAT_GENERATION_PROMPT = """You are a Senior SAP CPI QA Engineer.
Your task is to generate a UAT (User Acceptance Testing) Test Case specifically for the provided Variable Chain.

Context:
- Variable Name: {var_name}
- Scope: {var_scope}

The "Chain" represents the lifecycle of this variable: where it is created, used for decisions, modified, or persisted.

Step Descriptions (Reference):
{step_descriptions}

Variable Chain (Trace):
{chain_trace}

Task:
Generate 1-2 specific UAT Scenarios to verify the logic associated with this variable.
For each scenario, provide:
1. Scenario Name
2. Preconditions (Value of variable, setup required)
3. Input Data (If applicable)
4. Expected Behavior/Path (Narrative Format):
   - Instead of just listing IDs, explain the flow narratively.
   - Example: "First, the flow reaches Step 31 (Data Store), where it persists the payload..."
   - Explicitly mention the Step Type (e.g., Groovy Script, Content Modifier) and its specific action from the descriptions provided.
5. Success Criteria (What confirms the test passed?)

Format: Markdown. Do not include introductory filler text.
"""

class DeepSeekAIHelper:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = API_URL

    def _call_api(self, messages, json_mode=True):
        payload = {
            "model": MODEL,
            "messages": messages,
            "stream": False
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        
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
                    time.sleep(2 ** i)
                    continue
                return "[]" if json_mode else "(Rate Limit Error)"
            except Exception as e:
                print(f"API Error: {e}")
                return "[]" if json_mode else f"(Error: {str(e)})"
        return "[]" if json_mode else "(Max Retries Exceeded)"

    def extract_step_variables(self, context, data, process_name):
        user_msg = f"Process: {process_name}\nStep Type: {context}\nTechnical Data:\n{data}"
        messages = [{"role": "system", "content": VARIABLE_SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]
        response = self._call_api(messages, json_mode=True)
        try:
            # clean potential markdown code blocks if the LLM adds them
            clean_json = response.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except json.JSONDecodeError:
            return []

    def get_step_description(self, context, data, process_name):
        # Format the system prompt with the current process name
        system_msg = DESCRIPTION_SYSTEM_PROMPT.format(process_name=process_name)
        user_msg = f"Step Type: {context}\nTechnical Data:\n{data}"
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
        return self._call_api(messages, json_mode=False)

    def generate_uat(self, var_name, var_scope, chain_trace, step_descriptions):
        user_msg = UAT_GENERATION_PROMPT.format(
            var_name=var_name,
            var_scope=var_scope,
            chain_trace=chain_trace,
            step_descriptions=step_descriptions
        )
        messages = [{"role": "user", "content": user_msg}]
        return self._call_api(messages, json_mode=False)

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
        self.visited_nodes = set()
        
        # New Storage for Raw Step Data
        self.raw_step_data = [] 

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
            
            self.processes[pid] = {'start': best_start, 'name': pname, 'type': ptype}

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
        """
        Original DFS Logic Preserved.
        Changes: Calls extract_step_variables instead of get_step_summary.
        """
        if self.step_counter >= MAX_STEPS or node_id in self.visited_nodes:
            return

        node = self.elements.get(node_id)
        if not node: return

        if "Start" not in node['type']:
            self.visited_nodes.add(node_id)

        self.step_counter += 1
        tech_data = self.get_tech_data(node)
        
        print(f"    [{self.step_counter}] Analyzing {node['type']}...")
        
        # Call 1: Extract Variables (JSON)
        extracted_vars = self.ai.extract_step_variables(node['type'], tech_data, node['process_name'])
        
        # Call 2: Get Step Description (Text)
        step_description = self.ai.get_step_description(node['type'], tech_data, node['process_name'])
        
        # Store raw data for post-processing
        self.raw_step_data.append({
            "step_id": self.step_counter,
            "step_type": node['type'],
            "process": node['process_name'],
            "variables": extracted_vars if isinstance(extracted_vars, list) else [],
            "description": step_description
        })

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

    def _aggregate_and_structure_data(self):
        """
        Pivots the raw step-based data into the Variable Lifecycle Schema.
        """
        variables_map = {}

        for step in self.raw_step_data:
            step_id = step['step_id']
            step_type = step['step_type']
            step_desc = step.get('description', '')
            
            for var in step['variables']:
                v_name = var.get('name', 'Unknown')
                v_scope = var.get('scope', 'Unknown')
                
                # Normalize key
                key = f"{v_name}::{v_scope}"
                
                if key not in variables_map:
                    variables_map[key] = {
                        "variable_name": v_name,
                        "scope": v_scope,
                        "is_control_variable": False, 
                        "description": f"Variable {v_name} used in {step_type}", # Placeholder
                        "lifecycle_trace": [],
                        "usage_summary": {
                            "defined_at_steps": [],
                            "consumed_at_steps": [],
                            "controls_behavior_at_steps": [],
                            "persistence_points": []
                        }
                    }
                
                # Check for Control Variable Logic
                is_control = var.get('purpose') == 'DECISION'
                if is_control:
                    variables_map[key]['is_control_variable'] = True
                
                # Add to Lifecycle Trace with Description
                variables_map[key]['lifecycle_trace'].append({
                    "step_id": step_id,
                    "step_type": step_type,
                    "step_description": step_desc,
                    "action": var.get('action', 'UNKNOWN'),
                    "purpose": var.get('purpose', 'UNKNOWN'),
                    "context": var.get('context', '')
                })

                # Add to Usage Summary
                usage = variables_map[key]['usage_summary']
                if var.get('action') == 'CREATED':
                    usage['defined_at_steps'].append(step_id)
                elif var.get('action') in ['READ', 'MODIFIED']:
                    usage['consumed_at_steps'].append(step_id)
                
                if var.get('purpose') == 'DECISION':
                    usage['controls_behavior_at_steps'].append(step_id)
                if 'PERSISTENCE' in var.get('purpose', ''):
                    usage['persistence_points'].append(step_id)

        # Final Formatting
        final_var_list = list(variables_map.values())
        
        # Deduplicate summary lists
        for v in final_var_list:
            for k in v['usage_summary']:
                v['usage_summary'][k] = sorted(list(set(v['usage_summary'][k])))

        return {
            "analysis_metadata": {
                "flow_name": self.zip_path.stem,
                "total_variables_tracked": len(final_var_list),
                "generated_at": datetime.utcnow().isoformat() + "Z"
            },
            "variables": final_var_list
        }

    def generate_uat_report(self, variable_data_json):
        print("  [+] Generating UAT Scenarios for Variable Chains...")
        report_lines = [f"# UAT Report for {variable_data_json['analysis_metadata']['flow_name']}", ""]
        
        variables = variable_data_json['variables']
        
        # Filter variables to meaningful ones to avoid token exhaustion/time
        # Criteria: Controls logic OR Persisted OR Modified OR Used in Output OR Complete Chain
        
        for var in variables:
            # Skip if never consumed/used
            usage = var['usage_summary']
            is_interesting = (
                var['is_control_variable'] or 
                len(usage['persistence_points']) > 0 or
                len(usage['controls_behavior_at_steps']) > 0 or
                # Check if it connects at least two distinct steps (creation -> usage)
                (len(usage['defined_at_steps']) > 0 and len(usage['consumed_at_steps']) > 0)
            )
            
            if not is_interesting:
                continue

            var_name = var['variable_name']
            print(f"    > Generating UAT for chain: {var_name}")
            
            # 1. Build Chain Trace String
            chain_lines = []
            unique_step_ids = []
            step_descriptions_map = {} # id -> desc
            
            for trace_item in var['lifecycle_trace']:
                sid = trace_item['step_id']
                stype = trace_item['step_type']
                action = trace_item['action']
                context = trace_item['context']
                desc = trace_item.get('step_description', '')
                
                chain_lines.append(f"Step {sid} [{stype}]: {action} - {context}")
                
                # Deduplicate descriptions: only store if new step ID and description exists
                if sid not in step_descriptions_map and desc:
                    step_descriptions_map[sid] = desc
                    unique_step_ids.append(sid) # Keep order
            
            chain_text = "\n".join(chain_lines)
            
            # 2. Build Unique Descriptions String
            desc_lines = []
            for sid in unique_step_ids:
                desc_lines.append(f"Step {sid}: {step_descriptions_map[sid]}")
            desc_text = "\n\n".join(desc_lines)
            
            # 3. Call AI
            uat_content = self.ai.generate_uat(var_name, var['scope'], chain_text, desc_text)
            
            report_lines.append(f"## Variable: {var_name}")
            report_lines.append(uat_content)
            report_lines.append("\n---\n")
            
        return "\n".join(report_lines)

    def process(self):
        start_node = self.processes.get(self.main_process_id, {}).get('start')
        if not start_node:
            print("  [!] Could not find trigger in Participant_Process_1")
            return

        print(f"  [+] Tracing execution path for {self.zip_path.name}...")
        self.trace(start_node)

        print(f"  [+] Aggregating variable data...")
        final_json = self._aggregate_and_structure_data()
        
        # Save Variable Lifecycle JSON
        json_out = f"variable_lifecycle_{self.zip_path.stem}.json"
        with open(json_out, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, indent=2)
        print(f"  [OK] Variable Schema saved to: {json_out}")

        # Generate and Save UAT Report
        uat_report = self.generate_uat_report(final_json)
        report_out = f"uat_report_{self.zip_path.stem}.md"
        with open(report_out, 'w', encoding='utf-8') as f:
            f.write(uat_report)
        print(f"  [OK] UAT Report saved to: {report_out}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python cpi_variable_analyzer.py <file.zip>")
        return
    ai = DeepSeekAIHelper(DEEPSEEK_API_KEY)
    for zip_path in sys.argv[1:]:
        print(f"\n--- Batch Item: {zip_path} ---")
        analyzer = CPIFlowAnalyzer(zip_path, ai)
        if analyzer.load():
            analyzer.process()

if __name__ == "__main__":
    main()
