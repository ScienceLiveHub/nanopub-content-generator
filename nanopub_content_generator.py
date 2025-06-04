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
    system_prompt_template: str  # System instructions for format/structure
    max_length: int
    style_guidelines: str

class NanopubContentGenerator:
    def __init__(self, endpoint_url: str = "http://grlc.nanopubs.lod.labs.vu.nl"):
        self.endpoint_url = endpoint_url
        self.templates = self._initialize_templates()
        self.nanopub_endpoint = GrlcNanopubEndpoint(base_url=endpoint_url)
        
    def _initialize_templates(self) -> Dict[str, Template]:
        """Initialize predefined templates for different output formats"""
        return {
            "linkedin_post": Template(
                name="LinkedIn Post",
                structure="Hook + Context + Key Insights + Call to Action",
                system_prompt_template="""You are a professional content creator specializing in LinkedIn posts. Create a LinkedIn post based on the following nanopublication data:

Data: {content}
Vocabularies: {vocabularies}
Source Nanopublications: {sources}

SYSTEM REQUIREMENTS:
- Structure: Hook + Context + Key Insights + Call to Action
- Format: Professional, engaging, accessible to general audience
- Length: Under 3000 characters
- Include relevant hashtags
- No formal citations in the post content
- Use natural attribution (e.g., "Recent research shows...")

USER INSTRUCTIONS:
{user_instructions}

Generate the LinkedIn post following both the system requirements and user instructions above.""",
                max_length=3000,
                style_guidelines="Professional, engaging, accessible to general audience, no formal citations"
            ),
            
            "bluesky_post": Template(
                name="Bluesky Post",
                structure="Concise insight + Context + Citation",
                system_prompt_template="""You are a social media content creator for Bluesky. Create a Bluesky post based on this nanopublication:

Data: {content}
Vocabularies: {vocabularies}
Source Nanopublications: {sources}

SYSTEM REQUIREMENTS:
- Format: Concise but informative (under 300 characters)
- Include the key finding/insight
- Conversational tone
- Use relevant hashtags
- Keep it engaging and accessible

USER INSTRUCTIONS:
{user_instructions}

Generate the Bluesky post following both the system requirements and user instructions above.""",
                max_length=300,
                style_guidelines="Concise, conversational, informative"
            ),
            
            "opinion_paper": Template(
                name="Opinion Paper",
                structure="Abstract + Introduction + Main Arguments + Implications + Conclusion",
                system_prompt_template="""You are an academic writer creating an opinion paper. Write based on the nanopublication data:

Data: {content}
Vocabularies: {vocabularies}
Source Nanopublications: {sources}

SYSTEM REQUIREMENTS:
- Structure: Abstract + Introduction + Main Arguments + Implications + Conclusion
- Style: Academic but accessible
- Length: Around 2000 words
- Include proper reasoning and evidence-based arguments
- Maintain scholarly tone

USER INSTRUCTIONS:
{user_instructions}

Generate the opinion paper following both the system requirements and user instructions above.""",
                max_length=2000,
                style_guidelines="Academic but accessible, well-structured, evidence-based"
            ),
            
            "scientific_paper": Template(
                name="Scientific Paper Section",
                structure="Methods + Results + Discussion + References",
                system_prompt_template="""You are a scientific writer creating a paper section. Generate based on:

Data: {content}
Vocabularies: {vocabularies}
Source Nanopublications: {sources}

SYSTEM REQUIREMENTS:
- Structure: Methods + Results + Discussion
- Style: Precise, objective, technical
- Follow standard scientific writing conventions
- Use appropriate technical terminology
- Length: Around 1500 words

USER INSTRUCTIONS:
{user_instructions}

Generate the scientific paper section following both the system requirements and user instructions above.""",
                max_length=1500,
                style_guidelines="Precise, objective, technical, follows scientific conventions"
            )
        }
    
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
                content_item = {
                    'uri': data['uri'],
                    'title': parsed['publication_info'].get('title', 'Untitled'),
                    'author': parsed['publication_info'].get('author_name', 'Unknown'),
                    'summary': parsed['human_readable_summary'],
                    'main_claims': [stmt['human_readable'] for stmt in parsed['assertion_statements']],
                    'topics': parsed['topics_and_concepts'],
                    'entities': list(parsed['publication_info'].get('entity_labels', {}).values()),
                    'created': data['created']
                }
            else:
                # Fallback to raw assertion content (from nanopub-py method)
                content_item = {
                    'uri': data['uri'],
                    'assertion': data['assertion'][:500] + "..." if len(data['assertion']) > 500 else data['assertion'],
                    'created': data['created']
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
        
        # Format the system prompt with structured content and user instructions
        formatted_prompt = template.system_prompt_template.format(
            content=json.dumps(content_summary, indent=2),
            vocabularies=json.dumps(all_vocabularies, indent=2),
            sources="\n".join(sources),
            user_instructions=user_instructions
        )
        
        try:
            # Generate content using Ollama
            print(f"Calling Ollama with model: {model}")
            response = ollama.generate(
                model=model,
                prompt=formatted_prompt,
                options={
                    'temperature': 0.7,
                    'max_tokens': template.max_length // 4  # Rough token estimate
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
            print(f"Using template: {template.name}")
            
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
    parser.add_argument('--output', '-o', type=str,
                       help='Output file to save the result')
    
    args = parser.parse_args()
    
    # Initialize the generator
    generator = NanopubContentGenerator()
    
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
