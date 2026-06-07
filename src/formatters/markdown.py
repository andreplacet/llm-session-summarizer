from src.models import Message


class MarkdownFormatter:

    label = "📝 Markdown"

    def format_messages(self, messages: list[Message]) -> str:
        lines: list[str] = []
        for m in messages:
            role = "🔥 Desenvolvedor" if m.role == "user" else "🤖 IA"
            header = f"### {role}"
            ts = m.timestamp.strftime("%H:%M:%S") if m.timestamp else "??:??:??"
            lines.append(f"{header}  _{ts}_\n{m.text}")
        return "\n\n---\n\n".join(lines)

    def format_metadata(self, metadata: dict) -> str:
        return (
            f"- Início: {metadata.get('startTime', 'N/A')}\n"
            f"- Última atualização: {metadata.get('lastUpdated', 'N/A')}\n"
            f"- Fonte: {metadata.get('source', 'N/A')}"
        )
