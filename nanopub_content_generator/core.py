"""
Core content generation functionality.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from .endpoints import GrlcNanopubEndpoint
from .template_manager import TemplateManager
from .ai_interface import AIInterface


class NanopubContentGenerator:
    """Main class for generating content from nanopublications."""
    
    def __init__(self, endpoint_url: str = "http://grlc.nanopubs.lod.labs.vu.nl"):
        """Initialize the content generator."""
        self.endpoint = GrlcNanopubEndpoint(endpoint_url)
        self.template_manager = TemplateManager()
        self.ai_interface = AIInterface()
    
    async def run_pipeline(
        self,
        nanopub_uris: List[str],
        template_name: str,
        ollama_model: str = "llama3:8b",
        user_instructions: str = "",
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Run the complete content generation pipeline.
        
        Args:
            nanopub_uris: List of nanopublication URIs
            template_name: Name of the template to use
            ollama_model: Ollama model name
            user_instructions: Custom instructions for content generation
            description: Description of the generation task
            
        Returns:
            Dictionary containing generated content and metadata
        """
        try:
            # Step 1: Fetch nanopublications
            print(f"Fetching {len(nanopub_uris)} nanopublications...")
            nanopub_data = await self.fetch_multiple_nanopubs(nanopub_uris)
            
            if not nanopub_data:
                return {"error": "No nanopublications could be fetched"}
            
            # Step 2: Get template
            template = self.template_manager.get_template(template_name)
            if not template:
                return {"error": f"Template '{template_name}' not found"}
            
            # Step 3: Generate content
            generated_content = self.ai_interface.generate_content(
                template=template,
                nanopub_data=nanopub_data,
                model=ollama_model,
                user_instructions=user_instructions
            )
            
            # Step 4: Generate citations
            citations = self.generate_citation_list(nanopub_data)
            
            return {
                "template_used": template.name,
                "nanopubs_processed": len(nanopub_data),
                "generated_content": generated_content,
                "source_citations": citations,
                "metadata": {
                    "description": description,
                    "user_instructions": user_instructions,
                    "generated_at": datetime.now().isoformat(),
                    "model_used": ollama_model,
                    "nanopub_uris": nanopub_uris
                }
            }
            
        finally:
            await self.endpoint.close()
    
    async def fetch_multiple_nanopubs(self, nanopub_uris: List[str]) -> List[Dict[str, Any]]:
        """Fetch multiple nanopublications asynchronously."""
        tasks = [self.endpoint.fetch_nanopub(uri) for uri in nanopub_uris]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for result in results:
            if isinstance(result, dict) and "error" not in result:
                valid_results.append(result)
        
        return valid_results
    
    def generate_citation_list(self, nanopub_data: List[Dict[str, Any]]) -> str:
        """Generate a properly formatted citation list."""
        citations = []
        
        for i, data in enumerate(nanopub_data, 1):
            created_date = data.get('created', 'Unknown date')
            uri = data['uri']
            citation = f"[{i}] Nanopublication. {uri}. Retrieved {created_date}."
            citations.append(citation)
        
        return "\n".join(citations)


# Synchronous wrapper for backward compatibility
def run_pipeline_sync(*args, **kwargs):
    """Synchronous wrapper for the async run_pipeline method."""
    generator = NanopubContentGenerator()
    return asyncio.run(generator.run_pipeline(*args, **kwargs))
