from __future__ import annotations
import json
from typing import Dict, Any, Optional
from .base import Tool
from ..config import cfg
from ..config_manager import ConfigManager
from ..storage.sql_store import MemoryStore
from ..storage.faiss_index import FaissIndex
from ..retrieval import HybridRetriever
from ..block_builder import greedy_knapsack
from ..types import RetrievalQuery
from ..embedding import get_llama_embedder


class MemorySearchTool(Tool):
    """Tool for searching the memory database using the same process as the analyzer."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.store = MemoryStore(cfg.db_path)
        self.index = FaissIndex(cfg.index_path, cfg.embed_dim)
        self.retriever = HybridRetriever(self.store, self.index)
        self.embedder = get_llama_embedder()
    
    @property
    def name(self) -> str:
        return "memory_search"
    
    @property
    def description(self) -> str:
        return "Search the agent's memory database for relevant past experiences, conversations, and knowledge. Returns memories optimized for the available context window."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant memories"
                },
                "token_budget": {
                    "type": "integer",
                    "description": "Optional token budget for memory selection (defaults to context window minus reserves)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of memories to return (subject to token budget)",
                    "default": 20
                }
            },
            "required": ["query"]
        }
    
    def execute(self, query: str, token_budget: Optional[int] = None, max_results: int = 20) -> str:
        """
        Search memories using configured weights and knapsack algorithm.
        
        Args:
            query: Search query text
            token_budget: Optional token budget (defaults to context window - reserves)
            max_results: Maximum results to return
            
        Returns:
            JSON string containing search results
        """
        try:
            # Get current weights from config manager
            weights = self.config_manager.get_retrieval_weights()
            
            # Calculate token budget if not provided
            if token_budget is None:
                token_budget = cfg.context_window - cfg.reserve_output_tokens - cfg.reserve_system_tokens - 512
            token_budget = max(512, token_budget)
            
            # Decompose query using retriever's built-in method
            decomposition = self.retriever.decompose_query(query)
            
            # Create retrieval query
            rq = RetrievalQuery(
                session_id='tool_search',
                text=decomposition.get('what', query),
                actor_hint=decomposition['who'].get('id') if decomposition.get('who') else None,
                temporal_hint=decomposition.get('when')
            )
            
            # Get query embedding
            qvec = self.embedder.encode([query], normalize_embeddings=True)[0]
            
            # Search with configured weights - get many candidates for knapsack
            initial_candidates = 100  # Get more candidates than needed
            candidates = self.retriever.search_with_weights(
                rq, qvec, weights, 
                topk_sem=initial_candidates, 
                topk_lex=initial_candidates
            )
            
            # Apply knapsack algorithm to select memories within token budget
            selected_ids, tokens_used = greedy_knapsack(candidates, token_budget)
            
            # Filter to selected memories only
            selected_candidates = [c for c in candidates if c.memory_id in selected_ids]
            
            # Limit to max_results
            selected_candidates = selected_candidates[:max_results]
            
            # Format results for LLM consumption
            memories = []
            for candidate in selected_candidates:
                memory_data = self.store.get_memory(candidate.memory_id)
                if memory_data:
                    memories.append({
                        'memory_id': candidate.memory_id,
                        'score': round(candidate.score, 3),
                        'when': memory_data.get('when_str', ''),
                        'who': f"{memory_data.get('who_type', '')}: {memory_data.get('who_id', '')}",
                        'what': memory_data.get('what', ''),
                        'where': f"{memory_data.get('where_type', '')}: {memory_data.get('where_value', '')}",
                        'why': memory_data.get('why', ''),
                        'how': memory_data.get('how', ''),
                        'raw_text': memory_data.get('raw_text', '')[:500]  # Truncate for brevity
                    })
            
            result = {
                'success': True,
                'query': query,
                'memories_found': len(memories),
                'tokens_used': tokens_used,
                'token_budget': token_budget,
                'memories': memories
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'query': query
            }
            return json.dumps(error_result, indent=2)
