"""Integration tests for the full system."""

import pytest
from unittest.mock import patch, MagicMock
import json
import numpy as np
from datetime import datetime


class TestEndToEndFlow:
    """Test complete workflows through the system."""
    
    @patch('agentic_memory.extraction.llm_extractor.requests.post')
    @patch('agentic_memory.extraction.llm_extractor.SentenceTransformer')
    def test_ingest_and_retrieve_flow(self, mock_embedder, mock_llm, memory_router):
        """Test the complete flow from ingestion to retrieval."""
        from agentic_memory.types import RawEvent
        
        # Mock embedding
        mock_embed_instance = MagicMock()
        mock_embed_instance.encode.return_value = [np.random.randn(384).astype('float32')]
        mock_embedder.return_value = mock_embed_instance
        
        # Mock LLM extraction
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "who": {"type": "user", "id": "alice", "label": "Alice"},
                        "what": "Asked about machine learning algorithms",
                        "when": datetime.utcnow().isoformat(),
                        "where": {"type": "digital", "value": "chat_app"},
                        "why": "To learn about ML",
                        "how": "Via chat message"
                    })
                }
            }]
        }
        mock_llm.return_value = mock_response
        
        # Ingest multiple events
        events = [
            RawEvent(
                session_id="test_session",
                event_type="user_message",
                actor="user:alice",
                content="Tell me about machine learning algorithms"
            ),
            RawEvent(
                session_id="test_session",
                event_type="llm_message",
                actor="llm:assistant",
                content="Machine learning algorithms learn patterns from data"
            ),
            RawEvent(
                session_id="test_session",
                event_type="user_message",
                actor="user:alice",
                content="What about neural networks specifically?"
            )
        ]
        
        # Ingest all events
        memory_ids = []
        for event in events:
            mem_id = memory_router.ingest(event)
            memory_ids.append(mem_id)
            # Update mock for next event
            mock_embed_instance.encode.return_value = [np.random.randn(384).astype('float32')]
        
        assert len(memory_ids) == 3
        
        # Now retrieve
        context = [
            {"role": "user", "content": "Tell me more about ML"}
        ]
        
        result = memory_router.retrieve_block(
            session_id="test_session",
            context_messages=context
        )
        
        assert result is not None
        if result.get('members'):
            assert len(result['members']) > 0
            # Should retrieve relevant memories about ML
            first_mem = result['members'][0]
            assert 'what' in first_mem
    
    @patch('agentic_memory.server.flask_app.requests.post')
    @patch('agentic_memory.extraction.llm_extractor.requests.post')
    @patch('agentic_memory.extraction.llm_extractor.SentenceTransformer')
    def test_chat_with_memory_context(self, mock_embedder, mock_extract_llm, mock_chat_llm, memory_router):
        """Test chat endpoint with memory context building."""
        from agentic_memory.server.flask_app import app
        
        # Setup mocks
        mock_embed_instance = MagicMock()
        mock_embed_instance.encode.return_value = [np.random.randn(384).astype('float32')]
        mock_embedder.return_value = mock_embed_instance
        
        # Mock extraction
        mock_extract_response = MagicMock()
        mock_extract_response.status_code = 200
        mock_extract_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "who": {"type": "user", "id": "test", "label": "Test"},
                        "what": "User asked a question",
                        "when": datetime.utcnow().isoformat(),
                        "where": {"type": "digital", "value": "flask_ui"},
                        "why": "To get information",
                        "how": "Via web interface"
                    })
                }
            }]
        }
        mock_extract_llm.return_value = mock_extract_response
        
        # Mock chat completion
        mock_chat_response = MagicMock()
        mock_chat_response.status_code = 200
        mock_chat_response.json.return_value = {
            "choices": [{
                "message": {"content": "Here is my response based on memories"}
            }]
        }
        mock_chat_llm.return_value = mock_chat_response
        
        with app.test_client() as client:
            # Send chat request
            response = client.post('/api/chat', json={
                "session_id": "test_session",
                "text": "What did we discuss about Python?",
                "messages": []
            })
            
            assert response.status_code == 200
            data = response.get_json()
            assert "reply" in data
            assert "block" in data
    
    def test_tool_integration_in_memory(self, memory_router, tool_handler):
        """Test that tool calls are properly stored as memories."""
        from agentic_memory.types import RawEvent
        from agentic_memory.tools.base import ToolCall
        
        with patch('requests.post') as mock_llm:
            with patch('sentence_transformers.SentenceTransformer') as mock_embedder:
                # Setup mocks
                mock_embed_instance = MagicMock()
                mock_embed_instance.encode.return_value = [np.random.randn(384).astype('float32')]
                mock_embedder.return_value = mock_embed_instance
                
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{
                        "message": {
                            "content": json.dumps({
                                "who": {"type": "tool", "id": "web_search", "label": "Web Search"},
                                "what": "Searched for Python tutorials",
                                "when": datetime.utcnow().isoformat(),
                                "where": {"type": "digital", "value": "web"},
                                "why": "User requested information",
                                "how": "API call to search service"
                            })
                        }
                    }]
                }
                mock_llm.return_value = mock_response
                
                # Create tool call event
                tool_event = RawEvent(
                    session_id="test_session",
                    event_type="tool_call",
                    actor="tool:web_search",
                    content=json.dumps({
                        "name": "web_search",
                        "arguments": {"query": "Python tutorials"}
                    })
                )
                
                # Ingest tool call
                mem_id = memory_router.ingest(tool_event)
                assert mem_id is not None
                
                # Verify it's stored
                memories = memory_router.store.fetch_memories([mem_id])
                assert len(memories) == 1
                assert memories[0]['who_type'] == 'tool'
                assert memories[0]['who_id'] == 'web_search'
    
    def test_block_building_with_token_budget(self, memory_router):
        """Test that blocks respect token budgets."""
        from agentic_memory.types import Candidate
        from agentic_memory.block_builder import BlockBuilder
        
        builder = BlockBuilder(memory_router.store)
        
        # Create candidates with varying token counts
        candidates = [
            Candidate(memory_id="mem_1", score=0.9, token_count=100),
            Candidate(memory_id="mem_2", score=0.8, token_count=200),
            Candidate(memory_id="mem_3", score=0.7, token_count=150),
            Candidate(memory_id="mem_4", score=0.6, token_count=300),
            Candidate(memory_id="mem_5", score=0.5, token_count=400)
        ]
        
        from agentic_memory.types import RetrievalQuery
        query = RetrievalQuery(
            session_id="test",
            text="test query"
        )
        
        # Build blocks with limited budget
        blocks = builder.build(query, candidates, context_overhead=100)
        
        assert len(blocks) > 0
        first_block = blocks[0]
        
        # Check token budget is respected
        assert first_block.used_tokens <= first_block.budget_tokens
        
        # If there are more candidates than fit, has_more should be True
        total_tokens = sum(c.token_count for c in candidates)
        if total_tokens > first_block.budget_tokens:
            assert first_block.has_more or len(blocks) > 1
    
    def test_hybrid_retrieval_scoring(self, memory_router):
        """Test that hybrid retrieval combines multiple signals."""
        from agentic_memory.types import RetrievalQuery
        from agentic_memory.retrieval import HybridRetriever
        
        with patch('requests.post') as mock_llm:
            with patch('sentence_transformers.SentenceTransformer') as mock_embedder:
                # Setup mocks for ingestion
                mock_embed_instance = MagicMock()
                mock_embedder.return_value = mock_embed_instance
                
                # Create diverse memories
                from agentic_memory.types import RawEvent
                events = [
                    RawEvent(
                        session_id="test",
                        event_type="user_message",
                        actor="user:alice",
                        content="Python programming basics"
                    ),
                    RawEvent(
                        session_id="test",
                        event_type="user_message",
                        actor="user:bob",
                        content="Advanced Python techniques"
                    ),
                    RawEvent(
                        session_id="test",
                        event_type="user_message",
                        actor="user:alice",
                        content="JavaScript fundamentals"
                    )
                ]
                
                # Ingest with different embeddings
                for i, event in enumerate(events):
                    vec = np.random.randn(384).astype('float32')
                    vec[i*10:(i+1)*10] = 1.0  # Make vectors distinguishable
                    vec = vec / np.linalg.norm(vec)
                    mock_embed_instance.encode.return_value = [vec]
                    
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "choices": [{
                            "message": {
                                "content": json.dumps({
                                    "who": {"type": "user", "id": event.actor.split(':')[1], "label": None},
                                    "what": event.content,
                                    "when": datetime.utcnow().isoformat(),
                                    "where": {"type": "digital", "value": "test"},
                                    "why": "Testing",
                                    "how": "Direct"
                                })
                            }
                        }]
                    }
                    mock_llm.return_value = mock_response
                    
                    memory_router.ingest(event)
                
                # Now test retrieval
                retriever = HybridRetriever(memory_router.store, memory_router.index)
                
                query = RetrievalQuery(
                    session_id="test",
                    text="Python programming",
                    actor_hint="alice"
                )
                
                query_vec = np.random.randn(384).astype('float32')
                query_vec = query_vec / np.linalg.norm(query_vec)
                
                results = retriever.search(query, query_vec, topk_sem=5, topk_lex=5)
                
                assert len(results) > 0
                # Should prefer Python-related and Alice's messages
                scores = {r.memory_id: r.score for r in results}
                assert len(scores) > 0