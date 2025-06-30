"""
AI Interface for Content Generation

This module handles the integration with AI models (Ollama) and includes
the critical fix for the template echoing issue.
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

try:
    import ollama
except ImportError:
    print("Warning: ollama package not installed. Install with: pip install ollama")
    ollama = None


@dataclass
class Template:
    """Template data structure."""
    name: str
    structure: str
    content_guidelines: List[str]
    prompt_template: str
    max_length: int
    style_guidelines: str


class AIInterface:
    """
    Interface for AI model integration with proper content extraction.
    
    Key responsibility: Generate clean content without echoing template structure.
    """
    
    def __init__(self):
        """Initialize the AI interface."""
        if ollama is None:
            raise ImportError("Ollama package is required. Install with: pip install ollama")
    
    def generate_content(
        self,
        template: Template,
        nanopub_data: List[Dict[str, Any]],
        model: str = "llama3:8b",
        user_instructions: str = ""
    ) -> str:
        """
        Generate content using the AI model with proper content extraction.
        
        This is the main method that fixes the template echoing issue.
        """
        # Build the system prompt
        system_prompt = self._build_system_prompt(template, nanopub_data, user_instructions)
        
        # Generate content with AI
        try:
            response = ollama.generate(
                model=model,
                prompt=system_prompt,
                options={
                    'temperature': 0.3,  # Lower temperature for more factual output
                    'max_tokens': template.max_length // 3 if template.max_length > 2000 else template.max_length // 4
                }
            )
            
            ai_response = response.get('response', '')
            
            # CRITICAL: Extract clean content without template echoing
            clean_content = self._extract_clean_content(ai_response, template)
            
            return clean_content
            
        except Exception as e:
            raise RuntimeError(f"AI content generation failed: {e}")
    
    def _build_system_prompt(
        self,
        template: Template,
        nanopub_data: List[Dict[str, Any]],
        user_instructions: str
    ) -> str:
        """Build the system prompt for the AI model."""
        
        # Prepare nanopub content summary
        content_summary = self._prepare_nanopub_summary(nanopub_data)
        
        # Build sources list
        sources = [f"Nanopublication: {data['uri']}" for data in nanopub_data]
        
        # Enhanced user instructions
        if not user_instructions.strip():
            user_instructions = f"Create engaging {template.name.replace('_', ' ')} content that maintains scientific accuracy while being accessible to the target audience."
        
        # Format guidelines
        guidelines_text = "\n".join([f"- {guideline}" for guideline in template.content_guidelines])
        
        # Build the complete prompt with clear instructions to avoid template echoing
        system_prompt = f"""You are a professional content creator specializing in {template.name.replace('_', ' ')} content.

TASK: {template.prompt_template.format(user_instructions=user_instructions)}

DATA PROVIDED:
{content_summary}

SOURCE NANOPUBLICATIONS:
{chr(10).join(sources)}

STRUCTURE REQUIREMENTS:
{template.structure}

CONTENT GUIDELINES:
{guidelines_text}

CRITICAL REQUIREMENTS:
- Generate ONLY the final content, not explanations or meta-commentary
- Do NOT repeat or echo these instructions in your response
- Do NOT include template structure descriptions in your output
- ONLY use facts and claims that are explicitly stated in the provided data
- Maximum length: {template.max_length} characters
- Style: {template.style_guidelines}

IMPORTANT: Your response should contain ONLY the generated {template.name.replace('_', ' ')} content, nothing else. Do not include any prefatory text, explanations, or template structure descriptions.

Generate the content now:"""
        
        return system_prompt
    
    def _prepare_nanopub_summary(self, nanopub_data: List[Dict[str, Any]]) -> str:
        """Prepare a summary of nanopub data for the AI prompt."""
        if not nanopub_data:
            return "No nanopublication data available."
        
        summaries = []
        for i, data in enumerate(nanopub_data, 1):
            # Extract key information from nanopub data
            uri = data.get('uri', 'Unknown URI')
            
            # Try to get parsed content if available
            if 'parsed_content' in data and 'human_readable_summary' in data['parsed_content']:
                parsed = data['parsed_content']
                summary = f"Nanopub {i}:\nURI: {uri}\nSummary: {parsed['human_readable_summary']}"
                
                # Add main claims if available
                if parsed.get('assertion_statements'):
                    claims = [stmt['human_readable'] for stmt in parsed['assertion_statements'][:3]]
                    summary += f"\nKey Claims: {'; '.join(claims)}"
                
                # Add entities if available
                entity_labels = parsed.get('publication_info', {}).get('entity_labels', {})
                if entity_labels:
                    entities = list(entity_labels.values())[:3]
                    summary += f"\nEntities: {'; '.join(entities)}"
            
            # Fallback to assertion content
            elif 'assertion' in data and data['assertion']:
                assertion_preview = data['assertion'][:300] + "..." if len(data['assertion']) > 300 else data['assertion']
                summary = f"Nanopub {i}:\nURI: {uri}\nAssertion: {assertion_preview}"
            
            # Final fallback
            else:
                summary = f"Nanopub {i}:\nURI: {uri}\nContent: Limited parsing available"
            
            summaries.append(summary)
        
        return "\n\n".join(summaries)
    
    def _extract_clean_content(self, ai_response: str, template: Template) -> str:
        """
        Extract clean content from AI response, removing template echoing.
        
        This is the CRITICAL fix for the template echoing issue.
        """
        if not ai_response or not ai_response.strip():
            return f"Generated {template.name.replace('_', ' ')} content (empty response from AI)"
        
        content = ai_response.strip()
        
        # Remove common template echoing patterns
        content = self._remove_template_echoing(content, template)
        
        # Remove meta-commentary and instructions
        content = self._remove_meta_commentary(content)
        
        # Clean up formatting
        content = self._clean_formatting(content)
        
        # Validate minimum content length
        if len(content.strip()) < 50:
            # If content is too short, it might be an instruction echo
            fallback_content = self._generate_fallback_content(template, ai_response)
            return fallback_content
        
        return content.strip()
    
    def _remove_template_echoing(self, content: str, template: Template) -> str:
        """Remove template structure echoing from the content."""
        lines = content.split('\n')
        clean_lines = []
        
        # Patterns that indicate template echoing
        echo_patterns = [
            r'^(Template|Structure|Guidelines|Requirements|Task):',
            r'^(TASK|DATA PROVIDED|STRUCTURE REQUIREMENTS|CONTENT GUIDELINES|CRITICAL REQUIREMENTS):',
            r'^(You are a|Generate|Create a|Write a|The task is to):',
            r'^(Here is the|Here\'s the|Below is the):',
            r'^\*\*Template Used\*\*:',
            r'^\*\*Structure\*\*:',
            r'^\*\*Guidelines\*\*:',
        ]
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check if line matches any echo pattern
            is_echo = False
            for pattern in echo_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    is_echo = True
                    break
            
            # Skip lines that are template echoing
            if not is_echo:
                clean_lines.append(line)
        
        return '\n'.join(clean_lines)
    
    def _remove_meta_commentary(self, content: str) -> str:
        """Remove meta-commentary about the generation process."""
        # Remove common meta-commentary patterns
        meta_patterns = [
            r'Based on the (provided )?nanopublication data[,:]',
            r'Using the (provided )?information[,:]',
            r'From the nanopublication[s]?[,:]',
            r'According to the (provided )?data[,:]',
            r'The nanopublication[s]? (show|indicate|reveal)[s]?[,:]',
            r'This content is (based on|generated from)',
            r'(Here is|Here\'s) the (generated |requested )?content:',
            r'(Below is|Following is) the (generated |requested )?content:',
        ]
        
        lines = content.split('\n')
        clean_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if line is meta-commentary
            is_meta = False
            for pattern in meta_patterns:
                if re.search(pattern, line_stripped, re.IGNORECASE):
                    is_meta = True
                    break
            
            if not is_meta and line_stripped:
                clean_lines.append(line)
        
        return '\n'.join(clean_lines)
    
    def _clean_formatting(self, content: str) -> str:
        """Clean up formatting issues in the content."""
        # Remove excessive newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in content.split('\n')]
        content = '\n'.join(lines)
        
        # Remove leading/trailing whitespace
        content = content.strip()
        
        return content
    
    def _generate_fallback_content(self, template: Template, original_response: str) -> str:
        """Generate fallback content when extraction fails."""
        # Try to extract any meaningful content from the original response
        meaningful_lines = []
        
        for line in original_response.split('\n'):
            line = line.strip()
            # Keep lines that look like actual content (not instructions)
            if (line and 
                len(line) > 10 and 
                not line.startswith(('Task:', 'Template:', 'Generate', 'Create', 'Write')) and
                not line.isupper()):
                meaningful_lines.append(line)
        
        if meaningful_lines:
            return '\n'.join(meaningful_lines[:5])  # Take first 5 meaningful lines
        
        # Ultimate fallback
        return f"Generated {template.name.replace('_', ' ')} content based on provided nanopublication data. Content extraction encountered issues with the AI response formatting."
    
    def validate_model_availability(self, model: str) -> bool:
        """Check if the specified model is available in Ollama."""
        try:
            models = ollama.list()
            available_models = [m['name'] for m in models.get('models', [])]
            return model in available_models
        except Exception:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models in Ollama."""
        try:
            models = ollama.list()
            return [m['name'] for m in models.get('models', [])]
        except Exception:
            return []
