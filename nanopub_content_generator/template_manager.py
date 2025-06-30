"""
Template management system.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class Template:
    """Template for different output formats."""
    name: str
    structure: str
    content_guidelines: list
    prompt_template: str
    max_length: int
    style_guidelines: str


class TemplateManager:
    """Manages content generation templates."""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Template]:
        """Load templates from the templates directory."""
        templates = {}
        templates_dir = Path(__file__).parent.parent / "templates"
        
        if not templates_dir.exists():
            raise FileNotFoundError(f"Templates directory not found: {templates_dir}")
        
        for template_file in templates_dir.glob("*.json"):
            template_name = template_file.stem
            
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)
                
                template = Template(
                    name=template_data['name'],
                    structure=template_data['structure'],
                    content_guidelines=template_data['content_guidelines'],
                    prompt_template=template_data['prompt_template'],
                    max_length=template_data['max_length'],
                    style_guidelines=template_data['style_guidelines']
                )
                
                templates[template_name] = template
                
            except Exception as e:
                print(f"Warning: Could not load template {template_file}: {e}")
        
        return templates
    
    def get_template(self, template_name: str) -> Optional[Template]:
        """Get a template by name."""
        return self.templates.get(template_name)
    
    def list_templates(self) -> list:
        """List available template names."""
        return list(self.templates.keys())
