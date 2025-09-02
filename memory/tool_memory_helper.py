"""
Helper module for creating memories from tool use and function calls.
Supports integration with LLM tool-calling capabilities.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from memory.field_generator import FieldGenerator, MechanismType

logger = logging.getLogger(__name__)


class ToolMemoryHelper:
    """Helper for creating structured memories from tool interactions"""
    
    def __init__(self, memory_agent, field_generator: FieldGenerator):
        """
        Initialize the tool memory helper.
        
        Args:
            memory_agent: The memory agent instance
            field_generator: Field generator for intelligent field creation
        """
        self.memory_agent = memory_agent
        self.field_generator = field_generator
    
    def record_tool_use(self, 
                       tool_name: str,
                       tool_input: Dict[str, Any],
                       tool_output: Any,
                       who: str = "Assistant",
                       context: Optional[str] = None) -> bool:
        """
        Record a tool use as a memory.
        
        Args:
            tool_name: Name of the tool used
            tool_input: Input parameters to the tool
            tool_output: Output from the tool
            who: Who initiated the tool use
            context: Optional context about why the tool was used
            
        Returns:
            Success status
        """
        try:
            # Construct the 'what' field
            what = self._construct_tool_what(tool_name, tool_input, tool_output)
            
            # Generate 'why' field based on context
            if context:
                why = self.field_generator.generate_why_field(
                    context, who, "tool_use"
                )
            else:
                why = f"using {tool_name} tool"
            
            # Generate 'how' field
            how = self.field_generator.generate_how_field(
                MechanismType.TOOL_USE,
                {'tool_name': tool_name}
            )
            
            # Record the memory
            success, msg, event = self.memory_agent.remember(
                who=who,
                what=what,
                where="tool_execution",
                why=why,
                how=how,
                event_type="action"
            )
            
            if success:
                logger.info(f"Recorded tool use: {tool_name}")
            else:
                logger.error(f"Failed to record tool use: {msg}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error recording tool use: {e}")
            return False
    
    def record_function_call(self,
                            function_name: str,
                            arguments: Dict[str, Any],
                            result: Any,
                            who: str = "System",
                            error: Optional[str] = None) -> bool:
        """
        Record a function call as a memory.
        
        Args:
            function_name: Name of the function called
            arguments: Arguments passed to the function
            result: Result from the function
            who: Who initiated the call
            error: Optional error message if the call failed
            
        Returns:
            Success status
        """
        try:
            # Construct the 'what' field
            if error:
                what = f"Failed to call {function_name}: {error}"
            else:
                what = self._construct_function_what(function_name, arguments, result)
            
            # Generate 'why' field
            why = self.field_generator.generate_why_field(
                what, who, "function_call"
            )
            
            # Generate 'how' field
            how = self.field_generator.generate_how_field(
                MechanismType.FUNCTION_CALL,
                {'function': function_name}
            )
            
            # Record the memory
            success, msg, event = self.memory_agent.remember(
                who=who,
                what=what,
                where="function_execution",
                why=why,
                how=how,
                event_type="action" if not error else "observation"
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error recording function call: {e}")
            return False
    
    def record_web_search(self,
                         query: str,
                         results: List[Dict[str, Any]],
                         who: str = "Assistant") -> bool:
        """
        Record a web search as a memory.
        
        Args:
            query: The search query
            results: Search results
            who: Who initiated the search
            
        Returns:
            Success status
        """
        try:
            # Construct the 'what' field
            num_results = len(results) if results else 0
            what = f"Searched for '{query}' and found {num_results} results"
            
            # Generate 'why' field
            why = self.field_generator.generate_why_field(
                f"searching for {query}", who, "search"
            )
            
            # Generate 'how' field
            how = self.field_generator.generate_how_field(
                MechanismType.WEB_SEARCH,
                {'query': query}
            )
            
            # Record the memory
            success, msg, event = self.memory_agent.remember(
                who=who,
                what=what,
                where="web",
                why=why,
                how=how,
                event_type="action"
            )
            
            # Optionally record key results as observations
            if success and results:
                for i, result in enumerate(results[:3]):  # Top 3 results
                    self.memory_agent.observe(
                        what=f"Search result {i+1}: {result.get('title', '')}",
                        where="web",
                        link_to_action=event.id if event else None
                    )
            
            return success
            
        except Exception as e:
            logger.error(f"Error recording web search: {e}")
            return False
    
    def record_api_request(self,
                          endpoint: str,
                          method: str,
                          payload: Optional[Dict] = None,
                          response: Optional[Dict] = None,
                          status_code: Optional[int] = None,
                          who: str = "System") -> bool:
        """
        Record an API request as a memory.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            payload: Request payload
            response: Response data
            status_code: HTTP status code
            who: Who initiated the request
            
        Returns:
            Success status
        """
        try:
            # Construct the 'what' field
            what = f"{method} request to {endpoint}"
            if status_code:
                what += f" (status: {status_code})"
            
            # Generate 'why' field
            why = self.field_generator.generate_why_field(
                f"calling {endpoint}", who, "api_call"
            )
            
            # Generate 'how' field
            how = self.field_generator.generate_how_field(
                MechanismType.API_REQUEST,
                {'endpoint': endpoint}
            )
            
            # Record the memory
            success, msg, event = self.memory_agent.remember(
                who=who,
                what=what,
                where="api",
                why=why,
                how=how,
                event_type="action"
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error recording API request: {e}")
            return False
    
    def _construct_tool_what(self, tool_name: str, 
                            tool_input: Dict, 
                            tool_output: Any) -> str:
        """Construct a descriptive 'what' field for tool use"""
        # Summarize input
        input_summary = self._summarize_dict(tool_input, max_items=3)
        
        # Summarize output
        if isinstance(tool_output, dict):
            output_summary = self._summarize_dict(tool_output, max_items=2)
        elif isinstance(tool_output, list):
            output_summary = f"{len(tool_output)} items"
        else:
            output_summary = str(tool_output)[:100]
        
        return f"Used {tool_name} with {input_summary} → {output_summary}"
    
    def _construct_function_what(self, function_name: str,
                                arguments: Dict,
                                result: Any) -> str:
        """Construct a descriptive 'what' field for function calls"""
        # Summarize arguments
        args_summary = self._summarize_dict(arguments, max_items=3)
        
        # Summarize result
        if isinstance(result, dict):
            result_summary = self._summarize_dict(result, max_items=2)
        elif isinstance(result, list):
            result_summary = f"list of {len(result)} items"
        elif isinstance(result, bool):
            result_summary = "success" if result else "failure"
        else:
            result_summary = str(result)[:50]
        
        return f"Called {function_name}({args_summary}) → {result_summary}"
    
    def _summarize_dict(self, data: Dict, max_items: int = 3) -> str:
        """Create a brief summary of a dictionary"""
        if not data:
            return "no parameters"
        
        items = []
        for i, (key, value) in enumerate(data.items()):
            if i >= max_items:
                remaining = len(data) - max_items
                if remaining > 0:
                    items.append(f"...+{remaining} more")
                break
            
            if isinstance(value, str):
                value_str = f"'{value[:20]}...'" if len(value) > 20 else f"'{value}'"
            elif isinstance(value, (int, float, bool)):
                value_str = str(value)
            elif isinstance(value, list):
                value_str = f"[{len(value)} items]"
            elif isinstance(value, dict):
                value_str = f"{{...}}"
            else:
                value_str = type(value).__name__
            
            items.append(f"{key}={value_str}")
        
        return ", ".join(items)