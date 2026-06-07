import re
from typing import Optional

from src.models import Message, ParsedConversation
from src.parsers.base import AbstractParser


class OpenCodeMDParser(AbstractParser):
    """Parser for OpenCode .md session exports."""

    MSG_SEPARATOR = re.compile(r"\n---\n")

    THINKING_RE = re.compile(r"\n_Thinking:_[\s\S]*?(?=\n\*\*|\Z)", re.MULTILINE)

    TOOL_BLOCK_RE = re.compile(
        r"(?:^|\n)\*\*Tool:.*?\*\*\s*"
        r"(?:\n+\*\*Input:\*\*\s*\n```[\s\S]*?```\s*)?"
        r"(?:\n+\*\*Output:\*\*\s*\n```[\s\S]*?```\s*)?",
        re.MULTILINE,
    )

    TOOL_REM_LINE_RE = re.compile(r"^\*\*(Tool|Input|Output):.*?\*\*\s*$", re.MULTILINE)

    CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")

    def can_parse(self, raw) -> bool:
        if not isinstance(raw, str):
            return False
        return raw.startswith("# ") and "**Session ID:**" in raw[:500]

    def parse(self, raw) -> ParsedConversation:
        raw_str = raw if isinstance(raw, str) else ""

        metadata = self._parse_header(raw_str)
        metadata["source"] = "opencode"

        header_end = raw_str.find("\n---\n")
        if header_end == -1:
            return ParsedConversation(messages=[], metadata=metadata)

        content = raw_str[header_end + 5:]

        blocks = self.MSG_SEPARATOR.split(content)

        messages: list[Message] = []
        for block in blocks:
            msg = self._parse_block(block)
            if msg:
                messages.append(msg)

        return ParsedConversation(messages=messages, metadata=metadata)

    def _parse_header(self, raw: str) -> dict:
        metadata: dict = {}
        lines = raw.split("\n")

        if lines and lines[0].startswith("# "):
            metadata["title"] = lines[0][2:].strip()

        for line in lines:
            line = line.strip()
            if line.startswith("**Session ID:**"):
                metadata["sessionId"] = line.replace("**Session ID:**", "").strip()
            elif line.startswith("**Created:**"):
                metadata["startTime"] = line.replace("**Created:**", "").strip()
            elif line.startswith("**Updated:**"):
                metadata["lastUpdated"] = line.replace("**Updated:**", "").strip()
            elif line == "---":
                break

        return metadata

    def _parse_block(self, block: str) -> Optional[Message]:
        block = block.strip()
        if not block:
            return None

        first_line = block.split("\n", 1)[0].strip()

        if first_line.startswith("## Assistant"):
            role = "ai"
        elif first_line.startswith("## User"):
            role = "user"
        else:
            return None

        text = self._extract_text(block)
        if not text.strip():
            return None

        return Message(role=role, text=text.strip())

    def _extract_text(self, block: str) -> str:
        lines = block.split("\n")
        if lines and lines[0].startswith("## "):
            lines = lines[1:]

        text = "\n".join(lines)

        text = self.THINKING_RE.sub("", text)
        text = self.TOOL_BLOCK_RE.sub("", text)
        text = self.TOOL_REM_LINE_RE.sub("", text)
        text = self.CODE_FENCE_RE.sub("", text)

        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
