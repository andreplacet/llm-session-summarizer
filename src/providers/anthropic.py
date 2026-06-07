import os
from typing import AsyncIterator, Iterator, Optional

import anthropic

from src.providers.base import AbstractProvider

AVAILABLE_MODELS = [
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]

MODEL_LABELS = {
    "claude-opus-4-8": "Opus 4.8",
    "claude-sonnet-4-6": "Sonnet 4.6",
    "claude-haiku-4-5-20251001": "Haiku 4.5",
}

DEFAULT_MODEL = "claude-sonnet-4-6"
API_KEY_ENV = "ANTHROPIC_API_KEY"


class AnthropicProvider(AbstractProvider):

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        api_key = api_key or os.getenv(API_KEY_ENV)
        if not api_key:
            try:
                import streamlit as _st
                api_key = _st.secrets.get(API_KEY_ENV)
            except Exception:
                pass
        if not api_key:
            raise ValueError(
                f"{API_KEY_ENV} nao encontrada. Configure no .env, "
                "st.secrets (Cloud) ou passe explicitamente."
            )
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = await self._client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.content[0].text

    async def generate_stream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        async with self._client.messages.stream(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    def generate_stream_sync(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        client = anthropic.Anthropic(api_key=self._client.api_key)
        with client.messages.stream(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        ) as stream:
            for text in stream.text_stream:
                yield text
