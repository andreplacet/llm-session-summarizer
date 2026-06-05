from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Message:
    role: str
    text: str
    timestamp: Optional[datetime] = None


@dataclass
class ParsedConversation:
    messages: list[Message]
    metadata: dict
