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
        
        # Find all tool call patterns - try multiple formats for robustness
        patterns = [
            r'<tool_call>(.*?)</tool_call>',
            r'```tool_call\n(.*?)\n```',
            r'```json\n(.*?)\n```'  # Some models output as json code blocks
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, llm_response, re.DOTALL)
            
            for match in matches:
                try:
                    json_str = match.group(1).strip()
                    # Clean up common formatting issues
                    json_str = json_str.replace("'", '"')  # Replace single quotes
                    tool_data = json.loads(json_str)
                    
                    # Check if this looks like a tool call
                    if "name" in tool_data or ("tool" in tool_data and "arguments" in tool_data):
                        tool_name = tool_data.get("name") or tool_data.get("tool")
                        tool_calls.append(ToolCall(
                            name=tool_name,
                            arguments=tool_data.get("arguments", tool_data.get("params", {}))
                        ))
                        # Remove the matched tool call from response
                        cleaned_response = cleaned_response.replace(match.group(0), "")
                except (json.JSONDecodeError, AttributeError) as e:
                    logger.warning(f"Failed to parse potential tool call: {match.group(1)[:100]}, error: {e}")
        
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
                # Parse JSON results if it's from web_search
                if result.name == "web_search":
                    try:
                        search_data = json.loads(result.content)
                        formatted_text = f"[Web Search Results for: {search_data.get('query', '')}]\n\n"
                        for idx, res in enumerate(search_data.get('results', []), 1):
                            formatted_text += f"{idx}. {res.get('title', 'No title')}\n"
                            formatted_text += f"   {res.get('snippet', 'No description')}\n"
                            formatted_text += f"   Source: {res.get('link', '')}\n\n"
                        formatted.append(formatted_text)
                    except json.JSONDecodeError:
                        formatted.append(f"[TOOL_RESULT:{result.name}]\n{result.content}")
                else:
                    formatted.append(f"[TOOL_RESULT:{result.name}]\n{result.content}")
            else:
                formatted.append(f"[TOOL_ERROR:{result.name}]\n{result.error}")
        return "\n".join(formatted)

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
        tools_info = []
        for tool in self.tools.values():
            tools_info.append(f"- {tool.name}: {tool.description}")
        
        tools_list = "\n".join(tools_info)
        
        return f"""{base_prompt}

## Available Tools

You have access to the following tools to help answer questions:
{tools_list}

## Tool Usage Format

When you need to use a tool, you MUST output it in this EXACT format as the FIRST thing in your response:
<tool_call>{{"name": "web_search", "arguments": {{"query": "your search query"}}}}</tool_call>

## Examples

User: What's the weather today in Denver?
Assistant: <tool_call>{{"name": "web_search", "arguments": {{"query": "weather today Denver Colorado"}}}}</tool_call>

User: Tell me the latest news about AI
Assistant: <tool_call>{{"name": "web_search", "arguments": {{"query": "latest AI news today"}}}}</tool_call>

## Important Rules
1. For ANY question about current events, weather, news, recent information, or real-time data, you MUST use the web_search tool
2. Output the tool call FIRST before any explanation
3. Use valid JSON with double quotes (not single quotes)
4. Do NOT say "I cannot access current information" - use tools instead
5. Do NOT explain what you're about to do - just call the tool directly"""