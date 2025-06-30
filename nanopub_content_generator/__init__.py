"""
Nanopub Content Generator Package

Clean package-only implementation for AI-powered content generation from nanopublications.
No standalone scripts - everything accessed through proper Python imports.

Usage:
    from nanopub_content_generator import NanopubContentGenerator
    
    generator = NanopubContentGenerator()
    result = await generator.run_pipeline(
        nanopub_uris=["https://w3id.org/np/..."],
        template_name="linkedin_post",
        ollama_model="llama3:8b"
    )
"""

__version__ = "1.0.0"
__author__ = "Science Live Team"
__email__ = "support@sciencelive.com"
__description__ = "AI-powered content generation engine for nanopublications"

# Primary export - this is what the orchestration script imports
from .core import NanopubContentGenerator

# Additional exports for advanced usage
from .template_manager import TemplateManager
from .ai_interface import AIInterface
from .endpoints import GrlcNanopubEndpoint

# Convenience function for synchronous usage
from .core import run_pipeline_sync

__all__ = [
    "NanopubContentGenerator",  # Main class for orchestration
    "TemplateManager",          # Template handling
    "AIInterface",              # AI model integration
    "GrlcNanopubEndpoint",      # Nanopub data fetching
    "run_pipeline_sync"         # Sync wrapper if needed
]

# Package metadata
__package_info__ = {
    "name": "nanopub-content-generator",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "email": __email__,
    "repository": "https://github.com/ScienceLiveHub/nanopub-content-generator",
    "license": "MIT",
    "python_requires": ">=3.8"
}
