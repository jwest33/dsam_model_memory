"""
Web Search Tool using llama-cpp-agent's DuckDuckGo integration
Based on: https://github.com/Maximilian-Winter/llama-cpp-agent
"""
import json
from typing import Dict, Any
from .base import Tool

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDGS_AVAILABLE = True
    except ImportError:
        DDGS_AVAILABLE = False

try:
    from trafilatura import fetch_url, extract
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False


class LlamaAgentWebSearchTool(Tool):
    """Web search tool using llama-cpp-agent's DuckDuckGo integration."""
    
    def __init__(self, fetch_content: bool = False, max_results: int = 5):
        """
        Initialize the web search tool.
        
        Args:
            fetch_content: Whether to fetch and extract content from search result URLs
            max_results: Maximum number of search results to return
        """
        self.fetch_content = fetch_content
        self.max_results = max_results
        
        if not DDGS_AVAILABLE:
            raise ImportError("DuckDuckGo search not available. Install: pip install ddgs or duckduckgo-search")
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Search the web for information using DuckDuckGo. Can optionally fetch and extract content from result pages."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web"
                },
                "fetch_content": {
                    "type": "boolean",
                    "description": "Whether to fetch and extract content from result URLs (slower but more detailed)",
                    "default": self.fetch_content
                },
                "max_results": {
                    "type": "integer",
                    "description": f"Number of search results to return (default {self.max_results}, max 10)",
                    "default": self.max_results
                }
            },
            "required": ["query"]
        }
    
    def get_website_content(self, url: str) -> str:
        """
        Get website content from a URL using trafilatura.
        
        Args:
            url: URL to fetch content from
            
        Returns:
            Extracted content or error message
        """
        if not TRAFILATURA_AVAILABLE:
            return f"[Content extraction not available - install trafilatura]"
        
        try:
            downloaded = fetch_url(url)
            if not downloaded:
                return ""
            
            result = extract(downloaded, 
                           include_formatting=True, 
                           include_links=True, 
                           output_format='json', 
                           url=url)
            
            if result:
                data = json.loads(result)
                title = data.get("title", "No title")
                text = data.get("raw_text", "No content")
                
                # Truncate if too long
                if len(text) > 2000:
                    text = text[:2000] + "..."
                
                return f"Title: {title}\nContent: {text}"
            return ""
        except Exception as e:
            return f"[Error fetching content: {str(e)}]"
    
    def execute(self, query: str, fetch_content: bool = None, max_results: int = None) -> str:
        """
        Execute web search using DuckDuckGo.
        
        Args:
            query: The search query
            fetch_content: Override default content fetching setting
            max_results: Override default max results
            
        Returns:
            JSON string with search results
        """
        if fetch_content is None:
            fetch_content = self.fetch_content
        if max_results is None:
            max_results = self.max_results
        
        max_results = min(max_results, 10)
        
        try:
            # Perform search using DDGS
            with DDGS() as ddgs:
                # Using text search with parameters similar to the llama-agent example
                results = list(ddgs.text(
                    query, 
                    region='wt-wt',  # No specific region
                    safesearch='off',
                    max_results=max_results
                ))
            
            # Format results
            formatted_results = []
            for idx, result in enumerate(results[:max_results], 1):
                formatted_result = {
                    "position": idx,
                    "title": result.get("title", ""),
                    "snippet": result.get("body", ""),
                    "link": result.get("href", "")
                }
                
                # Optionally fetch page content
                if fetch_content and formatted_result["link"]:
                    content = self.get_website_content(formatted_result["link"])
                    if content:
                        formatted_result["content"] = content
                
                formatted_results.append(formatted_result)
            
            return json.dumps({
                "query": query,
                "results_count": len(formatted_results),
                "results": formatted_results
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Search failed: {str(e)}",
                "query": query,
                "results": []
            })