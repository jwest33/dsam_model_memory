from __future__ import annotations
from typing import Dict, Any
from pydantic import BaseModel

class ToolCall(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]

class ToolResult(BaseModel):
    tool_name: str
    output: Dict[str, Any] | str
