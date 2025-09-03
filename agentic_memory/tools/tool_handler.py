from __future__ import annotations
import json
import re
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)
from .base import Tool, ToolCall, ToolResult
from .web_search import WebSearchTool


class ToolHandler:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        self.register_tool(WebSearchTool())

    def register_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return [tool.to_function_def() for tool in self.tools.values()]

    def parse_tool_calls(self, llm_response: str) -> Tuple[str, List[ToolCall]]:
        """
        Parse LLM response for tool calls.
        Expected format: <tool_call>{"name": "web_search", "arguments": {"query": "..."}}</tool_call>
        Returns cleaned response and list of tool calls.
        """
        tool_calls = []
        cleaned_response = llm_response
        
        # Find all tool call patterns
        pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.finditer(pattern, llm_response, re.DOTALL)
        
        for match in matches:
            try:
                tool_data = json.loads(match.group(1))
                tool_calls.append(ToolCall(
                    name=tool_data.get("name"),
                    arguments=tool_data.get("arguments", {})
                ))
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in tool call: {match.group(1)[:100]}")
            # Always remove the tool call tags from response, even if invalid
            cleaned_response = cleaned_response.replace(match.group(0), "")
        
        # Clean up extra whitespace
        cleaned_response = re.sub(r'\n{3,}', '\n\n', cleaned_response).strip()
        
        return cleaned_response, tool_calls
    

    def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        if tool_call.name not in self.tools:
            return ToolResult(
                name=tool_call.name,
                content="",
                success=False,
                error=f"Unknown tool: {tool_call.name}"
            )
        
        tool = self.tools[tool_call.name]
        try:
            result_content = tool.execute(**tool_call.arguments)
            return ToolResult(
                name=tool_call.name,
                content=result_content,
                success=True
            )
        except Exception as e:
            return ToolResult(
                name=tool_call.name,
                content="",
                success=False,
                error=str(e)
            )

    def format_tool_message(self, tool_results: List[ToolResult]) -> str:
        """Format tool results for inclusion in LLM context."""
        formatted = []
        for result in tool_results:
            if result.success:
                formatted.append(f"[TOOL_RESULT:{result.name}]\n{result.content}")
            else:
                formatted.append(f"[TOOL_ERROR:{result.name}]\n{result.error}")
        return "\n\n".join(formatted)

    def should_use_tool(self, user_message: str) -> Optional[str]:
        """Check if user message requires tool use."""
        message_lower = user_message.lower()
        
        # Keywords that strongly suggest web search is needed
        search_triggers = [
            'current', 'today', 'now', 'latest', 'recent',
            'search', 'look up', 'find out', 'check',
            'weather', 'news', 'price', 'stock',
            'what is the date', 'what time',
        ]
        
        if any(trigger in message_lower for trigger in search_triggers):
            return 'web_search'
        
        return None
    
    def build_system_prompt_with_tools(self, base_prompt: str) -> str:
        """Build system prompt that includes tool usage instructions."""
        # If using OpenAI function calling, keep the prompt simple
        # The tool definitions are passed separately
        return f"""{base_prompt}

You have access to tools including web search.

IMPORTANT:
- Use the web_search tool when users ask for current, real-time, or factual information
- Do not claim you cannot access current information - use the tools provided
- Tool results will be provided to you automatically after calling them"""