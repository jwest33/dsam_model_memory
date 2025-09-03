"""Tests for 5W1H extraction."""

import pytest
from unittest.mock import patch, MagicMock
import json
from datetime import datetime


class TestExtraction:
    """Test suite for 5W1H extraction."""
    
    @patch('sentence_transformers.SentenceTransformer')
    @patch('requests.post')
    def test_extract_5w1h_user_message(self, mock_post, mock_embedder, sample_event):
        """Test extracting 5W1H from user message."""
        # Mock embedding
        import numpy as np
        mock_embed_instance = MagicMock()
        mock_embed_instance.encode.return_value = np.array([[0.1] * 384], dtype='float32')
        mock_embedder.return_value = mock_embed_instance
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "who": {"type": "user", "id": "test", "label": "Test User"},
                        "what": "User greeted and asked about Python",
                        "when": "2024-01-01T12:00:00",
                        "where": {"type": "digital", "value": "test_app"},
                        "why": "To start a conversation about programming",
                        "how": "Via text message"
                    })
                }
            }]
        }
        mock_post.return_value = mock_response
        
        from agentic_memory.extraction.llm_extractor import extract_5w1h
        
        result = extract_5w1h(sample_event, context_hint="Testing extraction")
        
        assert result.who.type == "user"
        assert result.who.id == "test"
        assert "Python" in result.what
        assert result.why == "To start a conversation about programming"
        assert result.how == "Via text message"
        assert result.raw_text == sample_event.content
        
        # Check embedding was called
        mock_embed_instance.encode.assert_called_once()
    
    @patch('sentence_transformers.SentenceTransformer')
    @patch('requests.post')
    def test_extract_5w1h_tool_call(self, mock_post, mock_embedder):
        """Test extracting 5W1H from tool call event."""
        from agentic_memory.types import RawEvent
        
        tool_event = RawEvent(
            session_id="test_session",
            event_type="tool_call",
            actor="tool:web_search",
            content=json.dumps({
                "name": "web_search",
                "arguments": {"query": "Python best practices"}
            }),
            metadata={"tool_type": "request"}
        )
        
        # Mock embedding
        import numpy as np
        mock_embed_instance = MagicMock()
        mock_embed_instance.encode.return_value = np.array([[0.1] * 384], dtype='float32')
        mock_embedder.return_value = mock_embed_instance
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "who": {"type": "tool", "id": "web_search", "label": "Web Search Tool"},
                        "what": "Tool called to search for Python best practices",
                        "when": datetime.utcnow().isoformat(),
                        "where": {"type": "digital", "value": "test_app"},
                        "why": "To find information about Python programming",
                        "how": "Via tool invocation"
                    })
                }
            }]
        }
        mock_post.return_value = mock_response
        
        from agentic_memory.extraction.llm_extractor import extract_5w1h
        
        result = extract_5w1h(tool_event)
        
        assert result.who.type == "tool"
        assert result.who.id == "web_search"
        assert "Python best practices" in result.what
    
    @patch('sentence_transformers.SentenceTransformer')
    @patch('requests.post')
    def test_extract_5w1h_with_context(self, mock_post, mock_embedder, sample_event):
        """Test that context hint is used in extraction."""
        # Mock embedding
        import numpy as np
        mock_embed_instance = MagicMock()
        mock_embed_instance.encode.return_value = np.array([[0.1] * 384], dtype='float32')
        mock_embedder.return_value = mock_embed_instance
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "who": {"type": "user", "id": "test", "label": None},
                        "what": "Discussion about code review",
                        "when": datetime.utcnow().isoformat(),
                        "where": {"type": "digital", "value": "test"},
                        "why": "Code review context",
                        "how": "Direct message"
                    })
                }
            }]
        }
        mock_post.return_value = mock_response
        
        from agentic_memory.extraction.llm_extractor import extract_5w1h
        
        context = "This is part of a code review discussion"
        result = extract_5w1h(sample_event, context_hint=context)
        
        # Verify context was included in LLM call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        request_body = call_args[1]['json']
        messages = request_body['messages']
        
        # Check that context appears in the messages
        full_text = ' '.join([msg['content'] for msg in messages])
        assert context in full_text
    
    @patch('sentence_transformers.SentenceTransformer')
    @patch('requests.post')
    def test_extract_5w1h_error_handling(self, mock_post, mock_embedder, sample_event):
        """Test error handling in extraction."""
        # Mock embedding
        import numpy as np
        mock_embed_instance = MagicMock()
        mock_embed_instance.encode.return_value = np.array([[0.1] * 384], dtype='float32')
        mock_embedder.return_value = mock_embed_instance
        
        # Mock LLM error
        mock_post.side_effect = Exception("LLM service unavailable")
        
        from agentic_memory.extraction.llm_extractor import extract_5w1h
        
        # Should fallback to basic extraction
        result = extract_5w1h(sample_event)
        
        assert "test" in result.who.id  # From actor field (may include prefix)
        assert result.what == sample_event.content
        assert result.raw_text == sample_event.content
        # Should have generated embedding even if LLM failed
        assert 'embed_vector_np' in result.extra
    
    @patch('sentence_transformers.SentenceTransformer')
    @patch('requests.post')
    def test_token_counting(self, mock_post, mock_embedder, sample_event):
        """Test that token counting works correctly."""
        # Mock embedding
        import numpy as np
        mock_embed_instance = MagicMock()
        mock_embed_instance.encode.return_value = np.array([[0.1] * 384], dtype='float32')
        mock_embedder.return_value = mock_embed_instance
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "who": {"type": "user", "id": "test", "label": None},
                        "what": "Test message",
                        "when": datetime.utcnow().isoformat(),
                        "where": {"type": "digital", "value": "test"},
                        "why": "Testing",
                        "how": "Direct"
                    })
                }
            }]
        }
        mock_post.return_value = mock_response
        
        from agentic_memory.extraction.llm_extractor import extract_5w1h
        
        result = extract_5w1h(sample_event)
        
        # Token count should be positive
        assert result.token_count > 0
        # Rough estimate: should be less than character count
        assert result.token_count < len(sample_event.content)
