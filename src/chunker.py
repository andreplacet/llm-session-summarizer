import asyncio

from src.models import ParsedConversation
from src.prompts.templates import CHUNK_PROMPT, MERGE_PROMPT, SYSTEM_PROMPT
from src.providers.base import AbstractProvider

CHUNK_SIZE = 10
CHUNK_OVERLAP = 2


def _build_conversation_text(conversation: ParsedConversation) -> str:
    lines: list[str] = []
    for msg in conversation.messages:
        role = "Desenvolvedor" if msg.role == "user" else "IA"
        lines.append(f"### {role}\n{msg.text}")
    return "\n\n---\n\n".join(lines)


async def summarize_conversation(
    provider: AbstractProvider,
    conversation: ParsedConversation,
    progress_callback=None,
) -> str:
    total = len(conversation.messages)

    if total == 0:
        return "Nenhuma mensagem encontrada na conversa."

    if total <= CHUNK_SIZE:
        text = _build_conversation_text(conversation)
        prompt = (
            f"Analise a conversa abaixo e gere um resumo estruturado.\n\n"
            f"**Metadados:** Sessão iniciada em {conversation.metadata.get('startTime', 'N/A')}\n\n"
            f"Conversa:\n{text}"
        )
        return await provider.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
        )

    chunks = _split_into_chunks(conversation)

    async def _summarize_chunk(idx: int, chunk: ParsedConversation) -> str:
        text = _build_conversation_text(chunk)
        result = await provider.generate(
            system_prompt="Você é um analista que resume conversas técnicas. Escreva em português do Brasil.",
            user_prompt=CHUNK_PROMPT.format(conversation_text=text),
        )
        return f"--- Trecho {idx + 1} de {len(chunks)} ---\n{result}"

    summaries = await asyncio.gather(
        *(_summarize_chunk(i, c) for i, c in enumerate(chunks))
    )

    if progress_callback:
        progress_callback(len(chunks), len(chunks))

    merge_prompt = MERGE_PROMPT.format(partial_summaries="\n\n".join(summaries))
    return await provider.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=merge_prompt,
    )


def _split_into_chunks(conversation: ParsedConversation) -> list[ParsedConversation]:
    messages = conversation.messages
    chunks: list[ParsedConversation] = []
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)

    for i in range(0, len(messages), step):
        chunk_msgs = messages[i : i + CHUNK_SIZE]
        chunks.append(
            ParsedConversation(messages=chunk_msgs, metadata=conversation.metadata)
        )

    return chunks
