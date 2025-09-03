#!/usr/bin/env python
"""Test web search tool functionality."""

import os
import json
from agentic_memory.tools import WebSearchTool
from agentic_memory.tools.tool_handler import ToolHandler
from agentic_memory.tools.base import ToolCall

def test_web_search_direct():
    """Test the web search tool directly."""
    print("Testing WebSearchTool directly...")
    
    # Create tool with or without API key
    tool = WebSearchTool()
    
    # Test search
    result = tool.execute(query="Python programming best practices", num_results=3)
    print("Direct search result:")
    print(json.dumps(json.loads(result), indent=2))
    print()

def test_tool_handler():
    """Test the tool handler with parsing."""
    print("Testing ToolHandler...")
    
    handler = ToolHandler()
    
    # Test tool definitions
    definitions = handler.get_tool_definitions()
    print("Available tools:")
    for defn in definitions:
        print(f"  - {defn['name']}: {defn['description']}")
    print()
    
    # Test parsing tool calls from LLM response
    llm_response = '''I'll search for information about Python best practices.

<tool_call>{"name": "web_search", "arguments": {"query": "Python programming best practices 2024", "num_results": 3}}</tool_call>

Let me look that up for you.'''
    
    cleaned, tool_calls = handler.parse_tool_calls(llm_response)
    print("Parsed response:")
    print(f"Cleaned text: {cleaned}")
    print(f"Tool calls found: {len(tool_calls)}")
    
    # Execute tool calls
    for tool_call in tool_calls:
        print(f"\nExecuting tool: {tool_call.name}")
        result = handler.execute_tool(tool_call)
        if result.success:
            print("Tool executed successfully")
            results = json.loads(result.content)
            if "error" in results:
                print(f"Tool returned error: {results['error']}")
            else:
                print(f"Found {len(results.get('results', []))} results")
        else:
            print(f"Tool execution failed: {result.error}")
    print()

def test_system_prompt():
    """Test system prompt generation."""
    print("Testing system prompt with tools...")
    
    handler = ToolHandler()
    base_prompt = "You are a helpful assistant."
    
    full_prompt = handler.build_system_prompt_with_tools(base_prompt)
    print("Generated system prompt:")
    print(full_prompt[:500] + "..." if len(full_prompt) > 500 else full_prompt)
    print()

if __name__ == "__main__":
    print("=" * 50)
    print("Web Search Tool Tests")
    print("=" * 50)
    print()
    
    # Note about API key
    if not os.getenv("SERPER_API_KEY") and not os.getenv("SEARCH_API_KEY"):
        print("NOTE: No API key found. Set SERPER_API_KEY or SEARCH_API_KEY environment variable")
        print("      for actual web searches. Tests will run but return error messages.")
        print()
    
    test_web_search_direct()
    test_tool_handler()
    test_system_prompt()
    
    print("=" * 50)
    print("Tests complete!")
    print("=" * 50)