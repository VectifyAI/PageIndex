from dataclasses import dataclass
from typing import Literal, Any


@dataclass
class QueryEvent:
    """Event emitted during streaming query."""
    type: Literal["reasoning", "tool_call", "tool_result", "text_delta", "text_done"]
    data: Any
