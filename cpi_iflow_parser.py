#!/usr/bin/env python3
"""
SAP CPI AI Explainer
1. Parses iFlow from ZIP.
2. Extracts technical logic (Groovy, Content Modifiers, Adapters).
3. Sends specific chunks to OpenRouter (using Google Gemma Free) for summarization.
4. limits execution to first 7 steps to save quota.
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
# PASTE YOUR OPENROUTER API KEY HERE
API_KEY = "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" 
# Using the specific free model requested
MODEL_NAME = "google/gemma-3n-e4b-it:free"
MAX_STEPS = 7  # Limit to first 7 steps
# ---------------------

NAMESPACES = {
    'bpmn2': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
    'ifl': 'http:///com.sap.ifl.model/Ifl.xsd',
}
for prefix, uri in NAMESPACES.items():
    ET.register_namespace(prefix, uri)

class AIHelper:
    def __init__(self, api_key):
        self.api_key = api_key
        # OpenRouter API Endpoint
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def generate_summary(self, prompt_context, data_content):
        """Sends text to OpenRouter and returns a 1-sentence summary."""
        if not data_content or not self.api_key or "PASTE_YOUR" in self.api_key:
            return "(Skipped: No API Key or Empty Content)"

        # Construct Prompt
        full_prompt = (
            f"You are an SAP Integration Expert. Summarize this {prompt_context} logic in ONE simple sentence. "
            f"Focus on the 'Why' and 'What', not the syntax.\n\n"
            f"DATA:\n{data_content}"
        )

        # OpenRouter / OpenAI Compatible Payload
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
        }

        # Headers for OpenRouter
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost", # Required by OpenRouter, using dummy
            "X-Title": "CPI Explainer Script"
        }

        # Retry Logic for Rate Limits (HTTP 429)
        max_retries = 3
        backoff_factor = 2 # Wait 2s, 4s, 8s...

        for attempt in range(max_retries + 1):
            try:
                req = urllib.request.Request(
                    self.url,
                    data=json.dumps(payload).encode('utf-8'),
                    headers=headers
                )
                with urllib.request.urlopen(req) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    # Extract text from OpenRouter/OpenAI format: choices[0].message.content
                    try:
                        return result['choices'][0]['message']['content'].strip()
                    except:
                        return "(AI Error: Could not parse response)"
            
            except urllib.error.HTTPError as e:
                # Handle Rate Limits (429)
                if e.code == 429:
                    if attempt < max_retries:
                        wait_time = backoff_factor ** (attempt + 1)
                        print(f"      (⚠️ Rate limit hit. Retrying in {wait_time}s...)")
                        time.sleep(wait_time)
                        continue
                    else:
                        return "(AI Error: Rate Limit Exceeded after retries)"
                
                # Try to read error body for more info
                try:
                    error_body = e.read().decode('utf-8')
                    print(f"DEBUG: {error_body}") 
                except:
                    pass
                return f"(AI Error: HTTP {e.code})"
            except Exception as e:
                return f"(AI Error: {str(e)})"

class CPIFlowAnalyzer:
    def __init__(self, zip_path):
        self.zip_path = Path(zip_path)
        self.xml_content = None
        self.root = None
        self.elements = {}
        self.step_counter = 0
        self.zip_file = None
        self.ai = AIHelper(API_KEY)

    def load(self):
        """Load and Parse XML"""
        try:
            self.zip_file = zipfile.ZipFile(self.zip_path, 'r')
            # Find .iflw
            candidates = [f for f in self.zip_file.namelist() if f.endswith('.iflw') and 'integrationflow' in f.lower()]
            if not candidates: raise Exception("No .iflw found")
            
            with self.zip_file.open(candidates[0]) as f:
                self.xml_content = f.read()
                
            self.root = ET.fromstring(self.xml_content)
            self._parse_elements()
        except Exception as e:
            print(f"Error loading ZIP: {e}")
            sys.exit(1)

    def _parse_elements(self):
        """Map all BPMN elements"""
        for elem in self.root.findall('.//*[@id]'):
            elem_id = elem.get('id')
            tag = elem.tag.split('}')[-1]
            props = self._get_ifl_properties(elem)
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
        
        if tag == 'endEvent':
            return "End"

        if tag == 'callActivity':
            if activity_type == 'Mapping':
                m_type = props.get('mappingType', '')
                if m_type == 'MessageMapping': return "Message Mapping"
                map_name = props.get('mappingname', '')
                if sub_activity_type == 'XSLTMapping' or 'XSLT' in map_name: return "XSLT Mapping"
                return "Mapping"
            if activity_type == 'Script':
                return "Groovy Script"
            if activity_type == 'Enricher':
                return "Content Modifier"
            if activity_type == 'Splitter':
                return "Splitter"
            if activity_type == 'XmlModifier':
                return "XML Modifier"
            if activity_type == 'ProcessCallElement':
                return "Process Call"
        
        if tag == 'serviceTask':
            if activity_type == 'ExternalCall':
                return "Request Reply"
            if activity_type == 'contentEnricherWithLookup':
                return "Content Enricher"
            if activity_type == 'Send':
                return "Send"

        if 'Gateway' in tag:
            return "Router"

        return activity_type if activity_type != 'Unknown' else tag

    def get_resource_content(self, filename):
        """Finds a file in the zip (script, mapping) and returns content"""
        if not filename: return None
        
        # Search patterns
        search_paths = [
            f"src/main/resources/script/{filename}",
            f"src/main/resources/groovy/{filename}",
            f"src/main/resources/mapping/{filename}",
            f"src/main/resources/xsd/{filename}"
        ]
        
        # Try direct match
        for path in search_paths:
            try:
                with self.zip_file.open(path) as f:
                    return f.read().decode('utf-8', errors='ignore')
            except KeyError:
                continue
                
        # Fallback: scan all files
        for f in self.zip_file.namelist():
            if f.endswith(filename):
                with self.zip_file.open(f) as file:
                    return file.read().decode('utf-8', errors='ignore')
        return None

    def analyze_node_with_ai(self, node):
        """Determines what content to send to AI based on node type"""
        name = node['name']
        props = node['props']
        n_type = node['type']
        
        ai_summary = ""
        
        # 1. GROOVY SCRIPTS
        if n_type == 'Groovy Script':
            script_file = props.get('script')
            if script_file:
                print(f"      (Reading {script_file}...)")
                content = self.get_resource_content(script_file)
                if content:
                    ai_summary = self.ai.generate_summary("Groovy Script", content)
                else:
                    ai_summary = "(File not found in ZIP)"

        # 2. CONTENT MODIFIER
        elif n_type == 'Content Modifier':
            # Extract headers/properties from XML config string (propertyTable)
            config_dump = f"Property Table: {props.get('propertyTable', '')}\nHeader Table: {props.get('headerTable', '')}"
            ai_summary = self.ai.generate_summary("Content Modifier Configuration", config_dump)

        # 3. REQUEST REPLY / EXTERNAL CALL
        elif n_type == 'Request Reply' or n_type == 'Content Enricher':
            # Dump adapter properties
            adapter_dump = json.dumps(props, indent=2)
            ai_summary = self.ai.generate_summary("Integration Adapter Configuration", adapter_dump)
            
        # 4. MAPPING
        elif "Mapping" in n_type:
            map_name = props.get('mappingname', 'Mapping')
            ai_summary = self.ai.generate_summary("Mapping Name context", f"Mapping Name: {map_name}. This is a transformation step.")

        return ai_summary

    def find_start_node(self):
        """Find Integration Process Start"""
        # Look for process named 'Integration Process'
        main_process = None
        for proc in self.root.findall('.//bpmn2:process', NAMESPACES):
            if 'Integration Process' in proc.get('name', ''):
                main_process = proc
                break
        
        if not main_process: return None
        start = main_process.find('bpmn2:startEvent', NAMESPACES)
        return start.get('id') if start is not None else None

    def run_limited_flow(self):
        """Traverse and AI-analyze first MAX_STEPS"""
        curr_id = self.find_start_node()
        if not curr_id:
            print("Could not find start node.")
            return

        print(f"\n🚀 Starting AI Analysis (First {MAX_STEPS} Steps) using OpenRouter...\n")
        
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
                print(f"   ✨ AI Summary: {summary}")
                # Rate limit protection: Free models can be slow or limited
                time.sleep(5) 
            # ----------------
            
            print("-" * 40)

            # Move next (Simple linear logic for demo)
            if node['outgoing']:
                # Prefer 'Main' path if multiple (heuristic)
                curr_id = node['outgoing'][0]['target']
            else:
                curr_id = None

        print(f"\n🛑 Stopped after {self.step_counter} steps (Test Mode).")

def main():
    if len(sys.argv) < 2:
        print("Usage: python cpi_ai_explainer.py <file.zip>")
        return
    
    analyzer = CPIFlowAnalyzer(sys.argv[1])
    analyzer.load()
    analyzer.run_limited_flow()

if __name__ == "__main__":
    main()
