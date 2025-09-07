"""Tests for tool handling system."""

import pytest
import json
from unittest.mock import patch, MagicMock


class TestToolHandler:
    """Test suite for tool handling."""
    
    def test_tool_registration(self, tool_handler):
        """Test that tools are registered correctly."""
        tools = tool_handler.tools
        assert 'web_search' in tools
        
        # Check tool definitions
        definitions = tool_handler.get_tool_definitions()
        assert len(definitions) >= 1
        
        web_search_def = next(d for d in definitions if d['name'] == 'web_search')
        assert 'description' in web_search_def
        assert 'parameters' in web_search_def
    
    def test_parse_tool_calls_single(self, tool_handler):
        """Test parsing a single tool call from LLM response."""
        llm_response = """I'll search for that information.
        
<tool_call>{"name": "web_search", "arguments": {"query": "Python async programming"}}</tool_call>

Let me look that up for you."""
        
        cleaned, tool_calls = tool_handler.parse_tool_calls(llm_response)
        
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "web_search"
        assert tool_calls[0].arguments["query"] == "Python async programming"
        
        # Check cleaned response
        assert "<tool_call>" not in cleaned
        assert "I'll search" in cleaned
        assert "Let me look" in cleaned
    
    def test_parse_tool_calls_multiple(self, tool_handler):
        """Test parsing multiple tool calls."""
        llm_response = """I'll search for multiple things.
        
<tool_call>{"name": "web_search", "arguments": {"query": "Python basics"}}</tool_call>

<tool_call>{"name": "web_search", "arguments": {"query": "JavaScript basics"}}</tool_call>

Found the information."""
        
        cleaned, tool_calls = tool_handler.parse_tool_calls(llm_response)
        
        assert len(tool_calls) == 2
        assert tool_calls[0].arguments["query"] == "Python basics"
        assert tool_calls[1].arguments["query"] == "JavaScript basics"
    
    def test_parse_tool_calls_none(self, tool_handler):
        """Test parsing response with no tool calls."""
        llm_response = "This is a regular response without any tool calls."
        
        cleaned, tool_calls = tool_handler.parse_tool_calls(llm_response)
        
        assert len(tool_calls) == 0
        assert cleaned == llm_response
    
    def test_parse_tool_calls_invalid_json(self, tool_handler):
        """Test handling invalid JSON in tool calls."""
        llm_response = """Response with bad tool call.
        
<tool_call>{invalid json here}</tool_call>

Rest of response."""
        
        cleaned, tool_calls = tool_handler.parse_tool_calls(llm_response)
        
        assert len(tool_calls) == 0
        # Tool call tags should be removed even if JSON is invalid
        assert "<tool_call>" not in cleaned
        assert "</tool_call>" not in cleaned
    
    @patch('agentic_memory.tools.web_search.requests.post')
    def test_execute_tool_success(self, mock_post, tool_handler):
        """Test successful tool execution."""
        from agentic_memory.tools.base import ToolCall
        
        # Mock DuckDuckGo unavailable to force Serper path
        with patch('agentic_memory.tools.web_search.DUCKDUCKGO_AVAILABLE', False):
            # Mock Serper API response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "organic": [
                    {"title": "Result 1", "snippet": "Test snippet", "link": "http://example.com"}
                ]
            }
            mock_post.return_value = mock_response
            
            # Set API key
            tool_handler.tools['web_search'].api_key = "test_key"
            tool_handler.tools['web_search'].backend = "serper"
            
            tool_call = ToolCall(
                name="web_search",
                arguments={"query": "test query"}
            )
            
            result = tool_handler.execute_tool(tool_call)
            
            assert result.success
            assert result.name == "web_search"
            data = json.loads(result.content)
            assert "results" in data
    
    def test_execute_tool_unknown(self, tool_handler):
        """Test executing unknown tool."""
        from agentic_memory.tools.base import ToolCall
        
        tool_call = ToolCall(
            name="unknown_tool",
            arguments={}
        )
        
        result = tool_handler.execute_tool(tool_call)
        
        assert not result.success
        assert "Unknown tool" in result.error
    
    def test_execute_tool_with_error(self, tool_handler):
        """Test tool execution with error."""
        from agentic_memory.tools.base import ToolCall
        
        # Force an error by passing invalid arguments
        with patch.object(tool_handler.tools['web_search'], 'execute', side_effect=Exception("Test error")):
            tool_call = ToolCall(
                name="web_search",
                arguments={"invalid": "args"}
            )
            
            result = tool_handler.execute_tool(tool_call)
            
            assert not result.success
            assert "Test error" in result.error
    
    def test_format_tool_message(self, tool_handler):
        """Test formatting tool results for LLM."""
        from agentic_memory.tools.base import ToolResult
        
        results = [
            ToolResult(
                name="web_search",
                content='{"results": [{"title": "Test"}]}',
                success=True
            ),
            ToolResult(
                name="failed_tool",
                content="",
                success=False,
                error="Connection failed"
            )
        ]
        
        formatted = tool_handler.format_tool_message(results)
        
        assert "[TOOL_RESULT:web_search]" in formatted
        assert "[TOOL_ERROR:failed_tool]" in formatted
        assert "Connection failed" in formatted
        assert '{"results"' in formatted
    
    def test_build_system_prompt_with_tools(self, tool_handler):
        """Test building system prompt with tool definitions."""
        base_prompt = "You are a helpful assistant."
        
        full_prompt = tool_handler.build_system_prompt_with_tools(base_prompt)
        
        assert base_prompt in full_prompt
        assert "You have access to the following tools:" in full_prompt
        assert "web_search" in full_prompt
        assert "<tool_call>" in full_prompt
        assert "arguments" in full_prompt


class TestLlamaAgentWebSearchTool:
    """Test suite for web search tool specifically."""
    
    def test_web_search_tool_available(self):
        """Test that LlamaAgentWebSearchTool is available."""
        from agentic_memory.tools.llama_agent_websearch import LlamaAgentWebSearchTool
        
        tool = LlamaAgentWebSearchTool()
        assert tool.name == "web_search"
    
    def test_web_search_parameters(self):
        """Test web search tool parameters."""
        from agentic_memory.tools.llama_agent_websearch import LlamaAgentWebSearchTool
        
        tool = LlamaAgentWebSearchTool()
        params = tool.parameters
        
        assert params["type"] == "object"
        assert "query" in params["properties"]
        assert "query" in params["required"]
