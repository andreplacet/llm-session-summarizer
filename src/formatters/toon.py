from src.models import Message


class ToonFormatter:

    label = "⚡ TOON (econômico)"

    def format_messages(self, messages: list[Message]) -> str:
        lines: list[str] = []
        for m in messages:
            role = "user" if m.role == "user" else "ai"
            header = f"role:{role}"
            if m.timestamp:
                header += f" ts:{m.timestamp.strftime('%H:%M:%S')}"
            lines.append(f"{header}\n{m.text}")
        return "\n\n".join(lines)

    def format_metadata(self, metadata: dict) -> str:
        parts: list[str] = []
        for key, label in [
            ("startTime", "start"),
            ("lastUpdated", "updated"),
            ("source", "source"),
        ]:
            val = metadata.get(key)
            if val:
                parts.append(f"{label}:{val}")
        return " ".join(parts)
