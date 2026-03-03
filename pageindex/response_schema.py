from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMResponse:
    content: str
    finish_reason: Optional[str] = None
    raw: Optional[dict] = None