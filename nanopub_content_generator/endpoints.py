import asyncio
import aiohttp
import nanopub
from typing import Dict, Any, Optional

class GrlcNanopubEndpoint():
    """Nanopub network via grlc API"""
    
    def __init__(self, base_url="http://grlc.nanopubs.lod.labs.vu.nl", timeout=30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = None
        
    async def _get_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self.session
    
    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def fetch_nanopub_rdf(self, uri: str, format: str = "trig") -> Dict[str, Any]:
        """
        Fetch nanopub in RDF format (trig, turtle, jsonld, etc.)
        Uses content negotiation to get RDF instead of HTML
        """
        session = await self._get_session()
        
        # Content negotiation headers for different RDF formats
        format_headers = {
            "trig": "application/trig",
            "turtle": "text/turtle", 
            "jsonld": "application/ld+json",
            "rdfxml": "application/rdf+xml",
            "nquads": "application/n-quads"
        }
        
        headers = {
            "Accept": format_headers.get(format, "application/trig")
        }
        
        try:
            async with session.get(uri, headers=headers) as response:
                content = await response.text()
                return {
                    'uri': uri,
                    'content': content,
                    'status': response.status,
                    'format': format,
                    'content_type': response.headers.get('content-type', 'unknown')
                }
        except Exception as e:
            return {'uri': uri, 'error': str(e)}
    
    def fetch_nanopub_with_nanopub_py(self, uri: str) -> Optional[Dict[str, Any]]:
        """
        Fetch nanopub using nanopub-py library (synchronous)
        This returns proper nanopub objects with assertion, provenance, etc.
        """
        try:
            # Try multiple approaches to handle different versions
            from nanopub import NanopubClient
            
            client = NanopubClient()
            
            # Try different methods that might be available
            publication = None
            error_msg = ""
            
            # Method 1: Try fetch() - from nanopub-py
            if hasattr(client, 'fetch'):
                try:
                    publication = client.fetch(uri)
                except Exception as e:
                    error_msg += f"fetch() failed: {e}; "
            
            # Method 2: Try download() - from some versions
            if publication is None and hasattr(client, 'download'):
                try:
                    publication = client.download(uri)
                except Exception as e:
                    error_msg += f"download() failed: {e}; "
            
            # Method 3: Try get() - alternative method
            if publication is None and hasattr(client, 'get'):
                try:
                    publication = client.get(uri)
                except Exception as e:
                    error_msg += f"get() failed: {e}; "
            
            if publication is None:
                available_methods = [m for m in dir(client) if not m.startswith('_')]
                return {
                    'uri': uri, 
                    'error': f'No working fetch method found. Tried: {error_msg}. Available methods: {available_methods}'
                }
            
            # Extract the different graphs as RDF strings
            result = {
                'uri': uri,
                'assertion': publication.assertion.serialize(format='turtle'),
                'provenance': publication.provenance.serialize(format='turtle'), 
                'pubinfo': publication.pubinfo.serialize(format='turtle'),
                'head': publication.head.serialize(format='turtle') if hasattr(publication, 'head') else '',
                'full_nanopub': publication.rdf.serialize(format='trig'),  # Complete nanopub in TriG
                'status': 200
            }
            
            return result
            
        except Exception as e:
            return {'uri': uri, 'error': str(e)}
    
    async def fetch_nanopub(self, uri: str, use_nanopub_py: bool = False, format: str = "trig") -> Dict[str, Any]:
        """
        Fetch nanopub - choose between nanopub-py, direct HTTP with content negotiation, or fallback methods
        """
        if use_nanopub_py:
            # Run the synchronous nanopub-py function in a thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.fetch_nanopub_with_nanopub_py, uri)
        else:
            # Use direct HTTP with content negotiation (more reliable)
            return await self.fetch_nanopub_rdf(uri, format)
    
    async def fetch_nanopub_json_ld(self, uri: str) -> Dict[str, Any]:
        """Convenience method to fetch as JSON-LD"""
        return await self.fetch_nanopub_rdf(uri, "jsonld")
    
    async def fetch_nanopub_turtle(self, uri: str) -> Dict[str, Any]:
        """Convenience method to fetch as Turtle"""
        return await self.fetch_nanopub_rdf(uri, "turtle")
