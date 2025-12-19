#!/usr/bin/env python3
"""
SAP CPI AI Explainer
1. Parses iFlow from ZIP.
2. Extracts technical logic (Groovy, Content Modifiers, Adapters).
3. Resolves external parameters (parameters.prop).
4. Sends optimized context to LOCAL OLLAMA API using Llama 3.1.
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
# LOCAL OLLAMA ENDPOINT
API_URL = "http://localhost:11434/api/chat"
# Using Local Model
MODEL_NAME = "llama3.1:8b" 
MAX_STEPS = 1000  # Limit to 1000 steps (effectively all)
# ---------------------

NAMESPACES = {
    'bpmn2': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
    'ifl': 'http:///com.sap.ifl.model/Ifl.xsd',
}
for prefix, uri in NAMESPACES.items():
    ET.register_namespace(prefix, uri)

class AIHelper:
    def __init__(self):
        self.url = API_URL

    def generate_summary(self, prompt_context, data_content):
        """Sends text to Local Ollama and returns a clear business summary."""
        if not data_content:
            return "(Skipped: Empty Content)"

        # Construct a specialized prompt for clarity
        full_prompt = (
            f"Explain this specific '{prompt_context}' step.\n"
            f"TECHNICAL DATA:\n{data_content}"
        )

        # Ollama Payload Structure
        payload = {
            "model": MODEL_NAME,
            "stream": False, # Important for simple Request/Response
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a concise SAP CPI expert. Summarize the logic in ONE to TWO simple sentence. Dont miss the things it used for that step for example it used x property for one thing in content modifier, x query in successfactors like that . Just say what it does."
                },
                {
                    "role": "user", 
                    "content": full_prompt
                }
            ]
        }

        try:
            req = urllib.request.Request(
                self.url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                try:
                    return result["message"]["content"].strip()
                except:
                    return "(AI Error: Could not parse response)"
        
        except urllib.error.URLError as e:
             return f"(AI Error: Connection Refused - Is Ollama running? {e})"
        except Exception as e:
            return f"(AI Error: {str(e)})"

class CPIFlowAnalyzer:
    def __init__(self, zip_path):
        self.zip_path = Path(zip_path)
        self.xml_content = None
        self.root = None
        self.elements = {}
        self.adapter_configs = {}  # Map node_id -> adapter properties
        self.parameters = {}       # Map param_key -> param_value
        self.step_counter = 0
        self.zip_file = None
        self.ai = AIHelper()

    def load(self):
        """Load and Parse XML"""
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
        """Parses parameters.prop to resolve {{placeholders}}"""
        try:
            candidates = [f for f in self.zip_file.namelist() if f.endswith('parameters.prop')]
            if not candidates: return
            with self.zip_file.open(candidates[0]) as f:
                content = f.read().decode('utf-8', errors='ignore')
                for line in content.splitlines():
                    if '=' in line and not line.startswith('#'):
                        key, val = line.split('=', 1)
                        self.parameters[key.strip()] = val.strip()
            print(f"   > Loaded {len(self.parameters)} external parameters.")
        except Exception:
            pass

    def _resolve_placeholders(self, props):
        """Replaces {{Key}} with actual value from parameters.prop"""
        resolved = {}
        for k, v in props.items():
            if v and isinstance(v, str) and '{{' in v and '}}' in v:
                clean_key = v.replace('{{', '').replace('}}', '').strip()
                if clean_key in self.parameters:
                    resolved[k] = self.parameters[clean_key]
                else:
                    resolved[k] = v 
            else:
                resolved[k] = v
        return resolved

    def _parse_elements(self):
        """Map all BPMN elements"""
        # Parse Message Flows (Adapters)
        for mf in self.root.findall('.//bpmn2:messageFlow', NAMESPACES):
            props = self._get_ifl_properties(mf)
            props = self._resolve_placeholders(props)
            source = mf.get('sourceRef')
            target = mf.get('targetRef')
            if source: self.adapter_configs[source] = props
            if target: self.adapter_configs[target] = props

        # Parse Nodes
        for elem in self.root.findall('.//*[@id]'):
            elem_id = elem.get('id')
            tag = elem.tag.split('}')[-1]
            props = self._get_ifl_properties(elem)
            props = self._resolve_placeholders(props)
            readable_type = self._determine_readable_type(elem, tag, props)
            
            self.elements[elem_id] = {
                'id': elem_id,
                'name': elem.get('name', ''),
                'tag': tag,
                'type': readable_type,
                'element': elem,
                'props': props,
                'outgoing': []
            }
            
        # Link Flows
        for sf in self.root.findall('.//bpmn2:sequenceFlow', NAMESPACES):
            source = sf.get('sourceRef')
            target = sf.get('targetRef')
            if source in self.elements and target in self.elements:
                self.elements[source]['outgoing'].append({
                    'target': target,
                    'condition': sf.find('bpmn2:conditionExpression', NAMESPACES).text if sf.find('bpmn2:conditionExpression', NAMESPACES) is not None else None
                })

    def _get_ifl_properties(self, elem):
        """Extract properties deeply"""
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
        """Determine specific step type"""
        activity_type = props.get('activityType', 'Unknown')
        sub_activity_type = props.get('subActivityType', '')
        
        if tag == 'startEvent':
            if elem.find('.//bpmn2:timerEventDefinition', NAMESPACES) is not None: return "Start Timer"
            if elem.find('.//bpmn2:messageEventDefinition', NAMESPACES) is not None: return "Start Message"
            return "Start"
        if tag == 'endEvent': return "End"
        if tag == 'callActivity':
            if activity_type == 'Mapping':
                map_name = props.get('mappingname', '')
                if sub_activity_type == 'XSLTMapping' or 'XSLT' in map_name: return "XSLT Mapping"
                return "Message Mapping"
            if activity_type == 'Script': return "Groovy Script"
            if activity_type == 'Enricher': return "Content Modifier"
            if activity_type == 'Splitter': return "Splitter"
            if activity_type == 'XmlModifier': return "XML Modifier"
            if activity_type == 'ProcessCallElement': return "Process Call"
            if activity_type == 'Filter': return "Filter"
            if activity_type == 'DBStorage': return "Data Store"
        
        if tag == 'serviceTask':
            if activity_type == 'ExternalCall': return "Request Reply"
            if activity_type == 'contentEnricherWithLookup': return "Content Enricher"
            if activity_type == 'Send': return "Send"
            if activity_type == 'DBStorage': return "Data Store"

        if 'Gateway' in tag: return "Router"
        return activity_type if activity_type != 'Unknown' else tag

    def get_resource_content(self, filename):
        """Finds a file in the zip and returns content"""
        if not filename: return None
        search_paths = [
            f"src/main/resources/script/{filename}",
            f"src/main/resources/groovy/{filename}",
            f"src/main/resources/mapping/{filename}",
            f"src/main/resources/xsd/{filename}"
        ]
        for path in search_paths:
            try:
                with self.zip_file.open(path) as f:
                    return f.read().decode('utf-8', errors='ignore')
            except KeyError: continue
        # Fallback scan
        for f in self.zip_file.namelist():
            if f.endswith(filename):
                with self.zip_file.open(f) as file:
                    return file.read().decode('utf-8', errors='ignore')
        return None

    def analyze_node_with_ai(self, node):
        """Determines what content to send to AI based on node type and gets summary"""
        name = node['name']
        props = node['props']
        n_type = node['type']
        node_id = node['id']
        
        debug_content = ""
        prompt_ctx = n_type
        
        # 1. GROOVY SCRIPTS
        if n_type == 'Groovy Script':
            script_file = props.get('script')
            if script_file:
                content = self.get_resource_content(script_file)
                if content:
                    debug_content = f"Script File: {script_file}\nContent Snippet: {content[:400]}...\n"
                    # Send full content to AI
                    return self.ai.generate_summary("Groovy Script", content)
                else:
                    return "(File not found in ZIP)"

        # 2. CONTENT MODIFIER
        elif n_type == 'Content Modifier':
            raw_prop = props.get('propertyTable') or ''
            raw_head = props.get('headerTable') or ''
            
            data_str = ""
            if 'Name' in raw_prop: 
                data_str += "Sets Properties: " + raw_prop.replace('<',' ').replace('>',' ') + "\n"
            if 'Name' in raw_head: 
                data_str += "Sets Headers: " + raw_head.replace('<',' ').replace('>',' ') + "\n"
            
            return self.ai.generate_summary("Content Modifier", data_str or "No variables set")

        # 3. ADAPTERS (Request Reply, Enricher, Send)
        elif n_type in ['Request Reply', 'Content Enricher', 'Send']:
            adapter_props = self.adapter_configs.get(node_id, {})
            full_props = {**props, **adapter_props}
            
            data_str = ""
            if 'resourcePath' in full_props: data_str += f"Entity: {full_props['resourcePath']}\n"
            if 'address' in full_props: data_str += f"System: {full_props['address']}\n"
            if 'queryOptions' in full_props: data_str += f"Query: {full_props['queryOptions']}\n"
            
            # Send simplified data + raw dump for context
            context = data_str + "\nRaw Config:\n" + json.dumps(full_props, indent=2)
            return self.ai.generate_summary("External Call/Enrichment", context)
            
        # 4. MAPPINGS
        elif "Mapping" in n_type:
            map_name = props.get('mappingname', 'Mapping')
            return self.ai.generate_summary("Mapping", f"Runs Mapping File: {map_name}")

        # 5. ROUTERS
        elif n_type == 'Router':
             return self.ai.generate_summary("Router", "This is a routing decision point.")

        # 6. SPLITTERS
        elif n_type == 'Splitter':
             expr = props.get('splitExprValue') or props.get('tokenValue') or ''
             return self.ai.generate_summary("Splitter", f"Splits message using expression: {expr}")

        return "" # Skip generic steps if no info

    def find_start_node(self):
        """Find Integration Process Start"""
        main_process = None
        for proc in self.root.findall('.//bpmn2:process', NAMESPACES):
            if 'Integration Process' in proc.get('name', ''):
                main_process = proc
                break
        if not main_process: return None
        start = main_process.find('bpmn2:startEvent', NAMESPACES)
        return start.get('id') if start is not None else None

    def run_limited_flow(self):
        """Traverse and AI-analyze Step-by-Step"""
        curr_id = self.find_start_node()
        if not curr_id:
            print("Could not find start node.")
            return

        print(f"\n🚀 Starting AI Analysis (Step-by-Step) using Local Ollama ({MODEL_NAME})...\n")
        
        visited = set()
        
        while curr_id and self.step_counter < MAX_STEPS:
            if curr_id in visited: break
            visited.add(curr_id)
            
            node = self.elements[curr_id]
            name = node['name']
            n_type = node['type']
            
            self.step_counter += 1
            print(f"{self.step_counter}. **{name}** ({n_type})")
            
            # --- AI MAGIC ---
            summary = self.analyze_node_with_ai(node)
            if summary:
                print(f"   ✨ AI: {summary}")
                # No rate limit needed for local LLM usually, but keeping a small pause is nice
                time.sleep(0.5) 
            # ----------------
            
            print("-" * 40)

            # Move next (Smart Heuristic)
            if node['outgoing']:
                outgoing_links = node['outgoing']
                next_id = outgoing_links[0]['target']
                if len(outgoing_links) > 1:
                    for link in outgoing_links:
                        target_node = self.elements.get(link['target'])
                        # Try to avoid immediate End events to keep flow going
                        if target_node and target_node['type'] != 'End':
                            next_id = link['target']
                            break
                curr_id = next_id
            else:
                curr_id = None

        print(f"\n🛑 Analysis Complete.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python cpi_ai_explainer.py <file.zip>")
        return
    
    analyzer = CPIFlowAnalyzer(sys.argv[1])
    analyzer.load()
    analyzer.run_limited_flow()

if __name__ == "__main__":
    main()
