from src.formatters.markdown import MarkdownFormatter
from src.formatters.toon import ToonFormatter

FORMATTERS: dict[str, "MarkdownFormatter | ToonFormatter"] = {
    "markdown": MarkdownFormatter(),
    "toon": ToonFormatter(),
}


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
