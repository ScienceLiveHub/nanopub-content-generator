#!/usr/bin/env python3
"""
Nanopublication Content Generator
Fetches nanopubs, applies templates, and generates human-readable content using Ollama
"""

import asyncio
import requests
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import nanopub
from rdflib import Graph, Namespace, URIRef
import ollama
from endpoints import GrlcNanopubEndpoint

@dataclass
class Template:
    """Template for different output formats"""
    name: str
    structure: str
    content_guidelines: List[str]
    prompt_template: str  # Simple template content, not full system prompt
    max_length: int
    style_guidelines: str

class NanopubContentGenerator:
    def __init__(self, endpoint_url: str = "http://grlc.nanopubs.lod.labs.vu.nl", templates_dir: str = "templates"):
        self.endpoint_url = endpoint_url
        self.templates_dir = templates_dir
        self.templates = self._load_templates()
        self.nanopub_endpoint = GrlcNanopubEndpoint(base_url=endpoint_url)
        
    def _load_templates(self) -> Dict[str, Template]:
        """Load templates from JSON files in the templates directory"""
        import os
        
        # Check if templates directory exists
        if not os.path.exists(self.templates_dir):
            raise FileNotFoundError(f"Templates directory '{self.templates_dir}' not found. Please create it and add template files.")
        
        templates = {}
        
        # Load all JSON files from templates directory
        for filename in os.listdir(self.templates_dir):
            if filename.endswith('.json'):
                template_name = filename[:-5]  # Remove .json extension
                file_path = os.path.join(self.templates_dir, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        template_data = json.load(f)
                    
                    # Validate required fields
                    required_fields = ['name', 'structure', 'content_guidelines', 'prompt_template', 'max_length', 'style_guidelines']
                    missing_fields = [field for field in required_fields if field not in template_data]
                    if missing_fields:
                        raise ValueError(f"Missing required fields: {missing_fields}")
                    
                    # Create Template object from JSON data
                    template = Template(
                        name=template_data['name'],
                        structure=template_data['structure'],
                        content_guidelines=template_data['content_guidelines'],
                        prompt_template=template_data['prompt_template'],
                        max_length=template_data['max_length'],
                        style_guidelines=template_data['style_guidelines']
                    )
                    
                    templates[template_name] = template
                    print(f"Loaded template: {template_name}")
                    
                except Exception as e:
                    raise RuntimeError(f"Error loading template from {filename}: {e}")
        
        if not templates:
            raise RuntimeError(f"No valid templates found in '{self.templates_dir}'. Please add template JSON files.")
        
        return templates
    
    def parse_trig_nanopub(self, trig_content: str, uri: str) -> Dict[str, Any]:
        """Parse TriG format nanopub and extract structured information"""
        try:
            from rdflib import Graph, Dataset
            from rdflib.namespace import RDF, RDFS, DCTERMS
            
            # Parse the TriG content using Dataset (replaces deprecated ConjunctiveGraph)
            dataset = Dataset()
            dataset.parse(data=trig_content, format='trig')
            
            # Extract the main components
            nanopub_info = {
                'uri': uri,
                'assertion_statements': [],
                'provenance_info': {},
                'publication_info': {},
                'entities_mentioned': [],
                'topics_and_concepts': [],
                'human_readable_summary': ''
            }
            
            # Find the main assertion content
            assertion_graph = None
            provenance_graph = None
            pubinfo_graph = None
            
            # Iterate through named graphs in the dataset
            for context_uri in dataset.contexts():
                if context_uri is None:
                    continue
                    
                context_id = str(context_uri.identifier)
                context_graph = dataset.get_context(context_uri.identifier)
                
                if 'assertion' in context_id:
                    assertion_graph = context_graph
                elif 'provenance' in context_id:
                    provenance_graph = context_graph
                elif 'pubinfo' in context_id:
                    pubinfo_graph = context_graph
            
            # Extract assertion information
            if assertion_graph:
                for s, p, o in assertion_graph:
                    statement = {
                        'subject': str(s),
                        'predicate': str(p),
                        'object': str(o),
                        'human_readable': self._make_human_readable_triple(s, p, o)
                    }
                    nanopub_info['assertion_statements'].append(statement)
                    
                    # Extract entities (things that are referenced)
                    if 'wikidata.org' in str(o):
                        nanopub_info['entities_mentioned'].append(str(o))
                    if 'schema.org/about' in str(p):
                        nanopub_info['topics_and_concepts'].append(str(o))
            
            # Extract provenance information
            if provenance_graph:
                for s, p, o in provenance_graph:
                    if 'wasAttributedTo' in str(p):
                        nanopub_info['provenance_info']['author'] = str(o)
                    nanopub_info['provenance_info'][str(p)] = str(o)
            
            # Extract publication info and labels
            if pubinfo_graph:
                labels = {}
                for s, p, o in pubinfo_graph:
                    if 'hasLabelFromApi' in str(p):
                        labels[str(s)] = str(o)
                    elif 'foaf:name' in str(p):
                        nanopub_info['publication_info']['author_name'] = str(o)
                    elif 'dct:created' in str(p):
                        nanopub_info['publication_info']['created'] = str(o)
                    elif 'rdfs:label' in str(p):
                        nanopub_info['publication_info']['title'] = str(o)
                
                nanopub_info['publication_info']['entity_labels'] = labels
            
            # Create human-readable summary
            nanopub_info['human_readable_summary'] = self._create_human_readable_summary(nanopub_info)
            
            return nanopub_info
            
        except Exception as e:
            print(f"Error parsing TriG content: {e}")
            return {
                'uri': uri,
                'error': f"Failed to parse TriG: {e}",
                'raw_content': trig_content[:500] + "..." if len(trig_content) > 500 else trig_content
            }
    
    def _make_human_readable_triple(self, s, p, o):
        """Convert RDF triple to human readable form"""
        # Simplify URIs to readable labels
        subj = self._simplify_uri(str(s))
        pred = self._simplify_uri(str(p))
        obj = self._simplify_uri(str(o))
        
        return f"{subj} {pred} {obj}"
    
    def _simplify_uri(self, uri):
        """Simplify URIs to more readable forms"""
        # Extract meaningful parts from URIs
        if 'wikidata.org/entity/' in uri:
            return f"[Wikidata Entity: {uri.split('/')[-1]}]"
        elif 'orcid.org/' in uri:
            return f"[ORCID: {uri.split('/')[-1]}]"
        elif 'doi.org/' in uri:
            return f"[DOI: {uri.split('doi.org/')[-1]}]"
        elif 'purl.org/aida/' in uri:
            # This contains the actual sentence - decode it
            import urllib.parse
            decoded = urllib.parse.unquote(uri.split('purl.org/aida/')[-1])
            return f"[Statement: {decoded[:100]}...]"
        elif '#' in uri:
            return uri.split('#')[-1]
        elif '/' in uri:
            parts = uri.split('/')
            return parts[-1] if parts[-1] else parts[-2]
        return uri
    
    def _create_human_readable_summary(self, nanopub_info):
        """Create a human-readable summary of the nanopublication"""
        summary_parts = []
        
        # Add title if available
        title = nanopub_info['publication_info'].get('title', '')
        if title:
            summary_parts.append(f"Title: {title}")
        
        # Add author information
        author_name = nanopub_info['publication_info'].get('author_name', '')
        if author_name:
            summary_parts.append(f"Author: {author_name}")
        
        # Add main statements
        if nanopub_info['assertion_statements']:
            summary_parts.append("Main Claims:")
            for i, stmt in enumerate(nanopub_info['assertion_statements'][:3], 1):  # Limit to first 3
                summary_parts.append(f"  {i}. {stmt['human_readable']}")
        
        # Add entity labels for context
        entity_labels = nanopub_info['publication_info'].get('entity_labels', {})
        if entity_labels:
            summary_parts.append("Referenced Entities:")
            for entity, label in list(entity_labels.items())[:5]:  # Limit to first 5
                clean_label = label.replace(' - ', ': ')
                summary_parts.append(f"  - {clean_label}")
        
        # Add creation date
        created = nanopub_info['publication_info'].get('created', '')
        if created:
            summary_parts.append(f"Created: {created}")
        
        return "\n".join(summary_parts)

    async def fetch_nanopublication(self, nanopub_uri: str) -> Optional[Dict[str, Any]]:
        """Fetch a single nanopublication from the endpoint"""
        try:
            # First try with HTTP content negotiation (more reliable)
            result = await self.nanopub_endpoint.fetch_nanopub(nanopub_uri, use_nanopub_py=False, format="trig")
            
            if 'error' in result:
                print(f"HTTP method failed, trying nanopub-py: {result['error']}")
                # Fallback to nanopub-py if HTTP fails
                result = await self.nanopub_endpoint.fetch_nanopub(nanopub_uri, use_nanopub_py=True)
                
                if 'error' in result:
                    print(f"Error fetching nanopublication {nanopub_uri}: {result['error']}")
                    return None
            
            # Process the result based on the method used
            if 'assertion' in result and 'provenance' in result and result['assertion']:
                # nanopub-py result - already structured
                graph_data = {
                    'assertion': result['assertion'],      
                    'provenance': result['provenance'],    
                    'pubinfo': result['pubinfo'],          
                    'head': result.get('head', ''),        
                    'full_nanopub': result.get('full_nanopub', result.get('content', '')), 
                    'uri': nanopub_uri,
                    'created': datetime.now().isoformat(),
                    'status': result['status']
                }
            else:
                # HTTP result - raw RDF content that needs to be parsed
                # Parse the TriG content to extract structured information
                parsed_nanopub = self.parse_trig_nanopub(result['content'], nanopub_uri)
                
                graph_data = {
                    'assertion': '',  # Raw assertion will be empty for HTTP method
                    'provenance': '', 
                    'pubinfo': '',    
                    'head': '',       
                    'full_nanopub': result['content'],  # Complete nanopub in TriG format
                    'parsed_content': parsed_nanopub,   # Structured, human-readable content
                    'uri': nanopub_uri,
                    'created': datetime.now().isoformat(),
                    'status': result['status'],
                    'format': result.get('format', 'trig')
                }
            
            return graph_data
            
        except Exception as e:
            print(f"Error fetching nanopublication {nanopub_uri}: {e}")
            return None
    
    async def fetch_multiple_nanopubs(self, nanopub_uris: List[str]) -> List[Dict[str, Any]]:
        """Fetch multiple nanopublications asynchronously"""
        tasks = [self.fetch_nanopublication(uri) for uri in nanopub_uris]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        valid_results = []
        for result in results:
            if isinstance(result, dict) and result is not None:
                valid_results.append(result)
            elif isinstance(result, Exception):
                print(f"Exception occurred: {result}")
        
        return valid_results
    
    def extract_vocabularies_and_ontologies(self, nanopub_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract controlled vocabularies and ontologies referenced in the nanopublication"""
        vocabularies = {
            'ontologies': [],
            'vocabularies': [],
            'namespaces': []
        }
        
        try:
            # Parse RDF content to extract vocabulary references
            content = nanopub_data.get('assertion', '') or nanopub_data.get('full_nanopub', '')
            if not content:
                return vocabularies
                
            g = Graph()
            if nanopub_data.get('format') == 'trig':
                g.parse(data=content, format='trig')
            else:
                g.parse(data=content, format='turtle')
            
            # Extract namespaces and ontologies
            for prefix, namespace in g.namespaces():
                vocabularies['namespaces'].append(f"{prefix}: {namespace}")
                
                # Common ontology patterns
                ns_str = str(namespace)
                if any(ont in ns_str.lower() for ont in ['owl', 'rdf', 'rdfs', 'skos', 'foaf', 'dcterms']):
                    vocabularies['ontologies'].append(ns_str)
                else:
                    vocabularies['vocabularies'].append(ns_str)
                    
        except Exception as e:
            print(f"Error extracting vocabularies: {e}")
            
        return vocabularies
    
    def generate_content_with_ollama(self, 
                                   template: Template, 
                                   nanopub_data: List[Dict[str, Any]], 
                                   user_instructions: str = "",
                                   model: str = "llama2") -> str:
        """Generate human-readable content using Ollama"""
        
        # Prepare content summary using parsed data when available
        content_summary = []
        all_vocabularies = {'ontologies': set(), 'vocabularies': set(), 'namespaces': set()}
        
        for data in nanopub_data:
            # Use parsed content if available (from HTTP method)
            if 'parsed_content' in data and 'human_readable_summary' in data['parsed_content']:
                parsed = data['parsed_content']
                
                # Extract more detailed information for scientific papers
                detailed_assertions = []
                for stmt in parsed['assertion_statements']:
                    detailed_assertions.append({
                        'subject': stmt['subject'],
                        'predicate': stmt['predicate'], 
                        'object': stmt['object'],
                        'human_readable': stmt['human_readable']
                    })
                
                content_item = {
                    'uri': data['uri'],
                    'title': parsed['publication_info'].get('title', 'Untitled'),
                    'author': parsed['publication_info'].get('author_name', 'Unknown'),
                    'summary': parsed['human_readable_summary'],
                    'main_claims': [stmt['human_readable'] for stmt in parsed['assertion_statements']],
                    'detailed_assertions': detailed_assertions,  # Full RDF details
                    'topics': parsed['topics_and_concepts'],
                    'entities': list(parsed['publication_info'].get('entity_labels', {}).values()),
                    'created': data['created'],
                    'raw_statements': [stmt for stmt in parsed['assertion_statements']]  # Include raw data for fact-checking
                }
            else:
                # Fallback to raw assertion content (from nanopub-py method)
                content_item = {
                    'uri': data['uri'],
                    'assertion': data['assertion'][:1000] + "..." if len(data['assertion']) > 1000 else data['assertion'],  # More content for scientific papers
                    'created': data['created'],
                    'note': 'Limited parsing - raw RDF content'
                }
                
                # Try to extract vocabularies from raw RDF
                vocabs = self.extract_vocabularies_and_ontologies(data)
                for key in all_vocabularies:
                    all_vocabularies[key].update(vocabs[key])
            
            content_summary.append(content_item)
        
        # Convert sets to sorted lists for consistent output
        for key in all_vocabularies:
            all_vocabularies[key] = sorted(list(all_vocabularies[key]))
        
        # Prepare source list (simpler than references)
        sources = [f"Nanopublication: {data['uri']}" for data in nanopub_data]
        
        # Use default user instructions if none provided
        if not user_instructions.strip():
            user_instructions = f"Create an engaging {template.name.lower()} that effectively communicates the research findings to the target audience."
        
        # Build the complete system prompt
        guidelines_text = "\n".join([f"- {guideline}" for guideline in template.content_guidelines])
        
        # For scientific papers, provide numbered citation mapping
        citation_info = ""
        if "scientific" in template.name.lower():
            citation_info = "\n\nCITATION REFERENCE MAPPING:\n"
            for i, data in enumerate(nanopub_data, 1):
                citation_info += f"[{i}] {data['uri']}\n"
            citation_info += f"\nUse these numbers [1], [2], etc. when citing specific nanopublications in your text."
            citation_info += f"\nIMPORTANT: Include ALL {len(nanopub_data)} references in your References section, not just the first few."
            citation_info += "\n\nSCIENTIFIC PAPER SPECIFIC INSTRUCTIONS:"
            citation_info += "\n- METHODS: Extract actual methodologies, experimental setups, or research approaches described in the nanopublications"
            citation_info += "\n- RESULTS: Present specific findings, measurements, or outcomes from the source data"
            citation_info += "\n- DISCUSSION: Analyze what the results mean based on what's stated in the nanopublications"
            citation_info += "\n- DO NOT write generic academic text - extract and synthesize the real research content\n"
        
        system_prompt = """You are a professional content creator specializing in {template_name}s.

DATA PROVIDED:
Data: {json.dumps(content_summary, indent=2)}
Vocabularies: {json.dumps(all_vocabularies, indent=2)}
Source Nanopublications: {'\n'.join(sources)}{citation_info}

TASK:
{template.prompt_template.format(user_instructions=user_instructions)}

STRUCTURE REQUIREMENTS:
{template.structure}

CONTENT GUIDELINES:
{guidelines_text}

CRITICAL REQUIREMENTS:
- ONLY use facts, claims, and information that are explicitly stated in the provided data
- DO NOT invent statistics, percentages, or specific numbers that are not in the source data
- If you mention specific claims, they must be directly traceable to the provided nanopublications
- When referencing findings, use general language like "research shows", "studies indicate", "evidence suggests" rather than inventing specific metrics
- Maximum length: {template.max_length} characters
- Style: {template.style_guidelines}

FACT-CHECKING REQUIREMENTS:
- Before writing any specific statistic, percentage, or number, verify it exists in the provided data
- If you cannot find a specific claim in the data, express it in general terms
- Example: Instead of "reduces time by 90%" say "significantly reduces time"
- Example: Instead of "improves efficiency by 50%" say "improves efficiency"
- When in doubt, be conservative and factual rather than specific and potentially incorrect

REMEMBER: Accuracy is more important than engagement. Do not fabricate data."""
        
        try:
            # Generate content using Ollama
            
            # Calculate max tokens more generously for longer content
            max_tokens = template.max_length // 3 if template.max_length > 2000 else template.max_length // 4
            
            response = ollama.generate(
                model=model,
                prompt=system_prompt,  # Use the built system prompt
                options={
                    'temperature': 0.3,  # Lower temperature for more factual, less creative output
                    'max_tokens': max_tokens
                }
            )
            
            return response['response']
            
        except Exception as e:
            print(f"Error generating content with Ollama (model: {model}): {e}")
            return f"Error generating content: {str(e)}"
    
    def generate_citation_list(self, nanopub_data: List[Dict[str, Any]]) -> str:
        """Generate a properly formatted citation list"""
        citations = []
        
        for i, data in enumerate(nanopub_data, 1):
            # Extract creation date and other metadata
            created_date = data.get('created', 'Unknown date')
            uri = data['uri']
            
            citation = f"[{i}] Nanopublication. {uri}. Retrieved {created_date}."
            citations.append(citation)
        
        return "\n".join(citations)
    
    async def run_pipeline(self, 
                          nanopub_uris: List[str], 
                          template_name: str, 
                          ollama_model: str = "llama2",
                          user_instructions: str = "",
                          description: str = "") -> Dict[str, Any]:
        """Run the complete pipeline"""
        
        print(f"Starting pipeline with {len(nanopub_uris)} nanopublications...")
        
        try:
            # Step 1: Fetch nanopublications
            print("Fetching nanopublications...")
            nanopub_data = await self.fetch_multiple_nanopubs(nanopub_uris)
            
            if not nanopub_data:
                return {"error": "No nanopublications could be fetched"}
            
            print(f"Successfully fetched {len(nanopub_data)} nanopublications")
            
            # Step 2: Get template
            if template_name not in self.templates:
                return {"error": f"Template '{template_name}' not found. Available: {list(self.templates.keys())}"}
            
            template = self.templates[template_name]
            
            # Step 3: Generate content
            generated_content = self.generate_content_with_ollama(
                template=template, 
                nanopub_data=nanopub_data, 
                user_instructions=user_instructions, 
                model=ollama_model
            )
            
            # Step 4: Generate citations
            print("Generating citations...")
            citations = self.generate_citation_list(nanopub_data)
            
            print("Pipeline completed successfully")
            # Return complete result
            return {
                "template_used": template.name,
                "nanopubs_processed": len(nanopub_data),
                "generated_content": generated_content,
                "source_citations": citations,  # Formal academic citations
                "metadata": {
                    "description": description,
                    "user_instructions": user_instructions,
                    "generated_at": datetime.now().isoformat(),
                    "model_used": ollama_model,
                    "nanopub_uris": nanopub_uris,
                    "note": "source_citations provide formal academic references for the nanopublications used"
                }
            }
            
        finally:
            # Always close the session
            await self.nanopub_endpoint.close()

# Example usage
async def main():
    import argparse
    import json
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Generate content from nanopublications')
    parser.add_argument('--config', '-c', type=str, 
                       help='Path to JSON config file with nanopub URIs and settings')
    parser.add_argument('--uris', '-u', nargs='+', 
                       help='Nanopub URIs (space-separated)')
    parser.add_argument('--template', '-t', type=str, 
                       choices=['linkedin_post', 'bluesky_post', 'opinion_paper', 'scientific_paper'],
                       help='Template to use for content generation')
    parser.add_argument('--model', '-m', type=str,
                       help='Ollama model to use')
    parser.add_argument('--templates-dir', type=str, default='templates',
                       help='Directory containing template JSON files (default: templates)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file to save the result')
    
    args = parser.parse_args()
    
    # Initialize the generator
    generator = NanopubContentGenerator(templates_dir=args.templates_dir)
    
    # Determine URIs to process
    nanopub_uris = []
    template_name = 'linkedin_post'  # Default
    model = 'llama2'  # Default
    
    if args.config:
        # Load from config file
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
                nanopub_uris = config.get('nanopub_uris', [])
                # Use config values as defaults
                template_name = config.get('template', 'linkedin_post')
                model = config.get('model', 'llama2')
                user_instructions = config.get('user_instructions', '')
                description = config.get('description', '')
                print(f"Loaded {len(nanopub_uris)} URIs from config file: {args.config}")
                print(f"Config template: {template_name}")
                print(f"Config model: {model}")
                if description:
                    print(f"Description: {description}")
                if user_instructions:
                    print(f"User instructions provided: {len(user_instructions)} characters")
                
                # CLI arguments override config values
                if args.template:
                    template_name = args.template
                    print(f"Overriding template with CLI argument: {template_name}")
                if args.model:
                    model = args.model
                    print(f"Overriding model with CLI argument: {model}")
                    
        except Exception as e:
            print(f"Error loading config file: {e}")
            return
    elif args.uris:
        # Use URIs from command line
        nanopub_uris = args.uris
        template_name = args.template or 'linkedin_post'
        model = args.model or 'llama2'
        user_instructions = ''
        description = ''
        print(f"Using {len(nanopub_uris)} URIs from command line")
    else:
        # Use default example
        nanopub_uris = [
            "https://w3id.org/np/RAJzZ8p6LBoe9D8ViX9DP2IIqZdxxfh-cQkBW3nfsYCzM",
        ]
        template_name = args.template or 'linkedin_post'
        model = args.model or 'llama2'
        user_instructions = ''
        description = ''
        print("Using default example URI")
    
    if not nanopub_uris:
        print("No nanopub URIs provided. Use --config or --uris")
        return
    
    print(f"Final configuration:")
    print(f"Template: {template_name}")
    print(f"Model: {model}")
    print(f"URIs to process: {len(nanopub_uris)}")
    if user_instructions:
        print(f"User instructions: {user_instructions[:100]}...")
    
    # Run pipeline
    result = await generator.run_pipeline(
        nanopub_uris=nanopub_uris,
        template_name=template_name,
        ollama_model=model,
        user_instructions=user_instructions,
        description=description
    )
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("Generated Content:")
        print("=" * 50)
        print(result["generated_content"])
        print("\n" + "=" * 50)
        print("Source Citations:")
        # Handle both old and new key names for backward compatibility
        citations = result.get("source_citations") or result.get("citations", "No citations available")
        print(citations)
        
        # Save to output file if specified
        if args.output:
            try:
                output_data = {
                    "generated_content": result["generated_content"],
                    "source_citations": citations,
                    "metadata": result["metadata"]
                }
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"\nResult saved to: {args.output}")
            except Exception as e:
                print(f"Error saving output: {e}")

if __name__ == "__main__":
    asyncio.run(main())
