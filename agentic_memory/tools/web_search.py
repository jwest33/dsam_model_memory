from __future__ import annotations
import requests
from typing import Any, Dict, List
import json
import os
from .base import Tool
try:
    from googlesearch import search as google_search
    GOOGLESEARCH_AVAILABLE = True
except ImportError:
    GOOGLESEARCH_AVAILABLE = False

try:
    from ddgs import DDGS  # Updated package name
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS  # Fallback to old package name
        DUCKDUCKGO_AVAILABLE = True
    except ImportError:
        DUCKDUCKGO_AVAILABLE = False


class WebSearchTool(Tool):
    def __init__(self, api_key: str = None, backend: str = "auto"):
        """
        Initialize web search tool.
        
        Args:
            api_key: Optional API key for paid services
            backend: Search backend to use ("auto", "duckduckgo", "google", "serper")
        """
        self.api_key = api_key or os.getenv("SERPER_API_KEY", "")
        if not self.api_key:
            self.api_key = os.getenv("SEARCH_API_KEY", "")
        self.base_url = "https://google.serper.dev/search"
        self.backend = backend
        
        # Auto-detect best available backend
        if backend == "auto":
            if DUCKDUCKGO_AVAILABLE:
                self.backend = "duckduckgo"
            elif GOOGLESEARCH_AVAILABLE:
                self.backend = "google"
            elif self.api_key:
                self.backend = "serper"
            else:
                self.backend = "duckduckgo"  # Will try to install if needed

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for information using Google search. Returns relevant search results including titles, snippets, and links."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of search results to return (default 5, max 10)",
                    "default": 5
                }
            },
            "required": ["query"]
        }

    def execute(self, query: str, num_results: int = 5) -> str:
        num_results = min(num_results, 10)
        
        # Try DuckDuckGo first (no API key needed)
        if self.backend == "duckduckgo":
            return self._search_duckduckgo(query, num_results)
        
        # Try Google search (no API key, but may have rate limits)
        elif self.backend == "google":
            return self._search_google(query, num_results)
        
        # Use Serper API if available
        elif self.backend == "serper" and self.api_key:
            return self._search_serper(query, num_results)
        
        # Fallback chain
        else:
            # Try free options first
            result = self._search_duckduckgo(query, num_results)
            if not json.loads(result).get("error"):
                return result
            
            result = self._search_google(query, num_results)
            if not json.loads(result).get("error"):
                return result
            
            # Finally try Serper if we have API key
            if self.api_key:
                return self._search_serper(query, num_results)
            
            return json.dumps({
                "error": "No search backend available. Install duckduckgo-search: pip install duckduckgo-search",
                "results": []
            })
    
    def _search_duckduckgo(self, query: str, num_results: int) -> str:
        """Search using DuckDuckGo (free, no API key required)."""
        if not DUCKDUCKGO_AVAILABLE:
            return json.dumps({
                "error": "DuckDuckGo search not available. Install: pip install duckduckgo-search",
                "results": []
            })
        
        try:
            results = []
            with DDGS() as ddgs:
                for idx, r in enumerate(ddgs.text(query, max_results=num_results)):
                    results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "link": r.get("href", "")
                    })
                    if idx + 1 >= num_results:
                        break
            
            return json.dumps({
                "query": query,
                "backend": "duckduckgo",
                "results": results
            })
        except Exception as e:
            return json.dumps({
                "error": f"DuckDuckGo search failed: {str(e)}",
                "results": []
            })
    
    def _search_google(self, query: str, num_results: int) -> str:
        """Search using googlesearch-python (free, but may have rate limits)."""
        if not GOOGLESEARCH_AVAILABLE:
            return json.dumps({
                "error": "Google search not available. Install: pip install googlesearch-python",
                "results": []
            })
        
        try:
            results = []
            for idx, url in enumerate(google_search(query, num=num_results, stop=num_results, pause=2)):
                # Note: googlesearch only returns URLs, not titles/snippets
                # We'd need to fetch pages for snippets, which is slow
                results.append({
                    "title": url.split('/')[2] if '/' in url else url,  # Use domain as title
                    "snippet": "Click to view content",
                    "link": url
                })
                if idx + 1 >= num_results:
                    break
            
            return json.dumps({
                "query": query,
                "backend": "google",
                "results": results
            })
        except Exception as e:
            return json.dumps({
                "error": f"Google search failed: {str(e)}",
                "results": []
            })
    
    def _search_serper(self, query: str, num_results: int) -> str:
        """Search using Serper API (paid service, best results)."""
        if not self.api_key:
            return json.dumps({
                "error": "No API key configured. Set SERPER_API_KEY or SEARCH_API_KEY environment variable.",
                "results": []
            })
        
        try:
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "q": query,
                "num": num_results
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract organic results
            results = []
            for item in data.get("organic", [])[:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", "")
                })
            
            # Add answer box if present
            answer_box = data.get("answerBox", {})
            if answer_box:
                results.insert(0, {
                    "title": "Answer Box",
                    "snippet": answer_box.get("snippet", answer_box.get("answer", "")),
                    "link": answer_box.get("link", "")
                })
            
            return json.dumps({
                "query": query,
                "backend": "serper",
                "results": results
            })
            
        except requests.exceptions.RequestException as e:
            return json.dumps({
                "error": f"Serper API request failed: {str(e)}",
                "results": []
            })
        except Exception as e:
            return json.dumps({
                "error": f"Unexpected error with Serper: {str(e)}",
                "results": []
            })