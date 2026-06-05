from datetime import datetime
from src.models import Message, ParsedConversation
from src.parsers.base import AbstractParser


class GeminiCLIParser(AbstractParser):

    def can_parse(self, raw: dict) -> bool:
        return (
            isinstance(raw, dict)
            and "sessionId" in raw
            and "kind" in raw
            and "messages" in raw
        )

    def parse(self, raw: dict) -> ParsedConversation:
        messages: list[Message] = []

        for msg in raw.get("messages", []):
            msg_type = msg.get("type", "")
            timestamp = self._parse_timestamp(msg.get("timestamp"))

            if msg_type == "info":
                continue

            if msg_type == "user":
                text = self._extract_user_text(msg)
                if text:
                    messages.append(Message(role="user", text=text, timestamp=timestamp))

            elif msg_type == "gemini":
                text = self._extract_gemini_text(msg)
                if text:
                    messages.append(Message(role="ai", text=text, timestamp=timestamp))

        metadata = {
            "sessionId": raw.get("sessionId"),
            "startTime": raw.get("startTime"),
            "lastUpdated": raw.get("lastUpdated"),
            "kind": raw.get("kind"),
            "source": "gemini_cli",
        }

        return ParsedConversation(messages=messages, metadata=metadata)

    def _parse_timestamp(self, ts: str | None) -> datetime | None:
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    def _extract_user_text(self, msg: dict) -> str:
        content = msg.get("content")

        if isinstance(content, str):
            text = content.strip()
            if self._is_skippable(text):
                return ""
            return text

        if not isinstance(content, list):
            return ""

        if self._has_function_response(content):
            return ""

        texts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            t = item.get("text", "")
            if isinstance(t, str):
                t = t.strip()
                if not self._is_skippable(t):
                    texts.append(t)

        return "\n".join(texts)

    def _extract_gemini_text(self, msg: dict) -> str:
        parts: list[str] = []

        thoughts = msg.get("thoughts")
        if isinstance(thoughts, list) and thoughts:
            thought_texts = []
            for t in thoughts:
                desc = t.get("description", "") if isinstance(t, dict) else ""
                if desc:
                    thought_texts.append(desc)
            if thought_texts:
                parts.append("**Raciocínio:**\n" + "\n".join(f"- {t}" for t in thought_texts))

        content = msg.get("content", "")
        if content and isinstance(content, str) and content.strip():
            parts.append(content.strip())

        tool_calls = msg.get("toolCalls")
        if isinstance(tool_calls, list) and tool_calls:
            tool_names: set[str] = set()
            for tc in tool_calls:
                if isinstance(tc, dict) and "name" in tc:
                    tool_names.add(tc["name"])
            if tool_names:
                parts.append(f"\n*(Ferramentas: {', '.join(sorted(tool_names))})*")

        return "\n\n".join(parts)

    def _has_function_response(self, content: list) -> bool:
        for item in content:
            if isinstance(item, dict) and "functionResponse" in item:
                return True
        return False

    def _is_skippable(self, text: str) -> bool:
        return any([
            text.startswith("<session_context>"),
            text.startswith("Here is the user's editor context"),
            text.startswith("<system-reminder>"),
        ])
