import os
from typing import AsyncIterator, Iterator, Optional

from openai import AsyncOpenAI

from src.providers.base import AbstractProvider

AVAILABLE_MODELS = [
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
]

MODEL_LABELS = {
    "gpt-5.5": "GPT-5.5",
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4-mini": "GPT-5.4 Mini",
}

DEFAULT_MODEL = "gpt-5.5"
API_KEY_ENV = "OPENAI_API_KEY"


class OpenAIProvider(AbstractProvider):

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
    ):
        self.model = model
        self.temperature = temperature
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
        self._client = AsyncOpenAI(api_key=api_key)

    def _messages(self, system_prompt: str, user_prompt: str) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=self._messages(system_prompt, user_prompt),
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    async def generate_stream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=self._messages(system_prompt, user_prompt),
            temperature=self.temperature,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def generate_stream_sync(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        from openai import OpenAI
        client = OpenAI(api_key=self._client.api_key)
        stream = client.chat.completions.create(
            model=self.model,
            messages=self._messages(system_prompt, user_prompt),
            temperature=self.temperature,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
