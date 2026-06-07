import os
from typing import AsyncIterator, Optional

from google import genai

from src.providers.base import AbstractProvider

AVAILABLE_MODELS = [
    "gemini-2.0-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
]

MODEL_LABELS = {
    "gemini-2.0-flash": "🆓 Grátis (free tier)",
    "gemini-3-flash-preview": "💰 Pago (preview)",
    "gemini-3.1-pro-preview": "💰 Pago (preview)",
}

MODEL_ALIASES = {
    "flash": "gemini-2.0-flash",
    "pro": "gemini-3.1-pro-preview",
    "flash-3": "gemini-3-flash-preview",
}

DEFAULT_MODEL = "gemini-2.0-flash"


class GeminiProvider(AbstractProvider):

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
    ):
        self.model = model
        self.temperature = temperature
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            try:
                import streamlit as _st
                api_key = _st.secrets.get("GEMINI_API_KEY")
            except Exception:
                pass
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY não encontrada. Configure no .env, "
                "st.secrets (Cloud) ou passe explicitamente."
            )
        self._client = genai.Client(api_key=api_key)

    def _build_config(self, system_prompt: str) -> genai.types.GenerateContentConfig:
        return genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
        )

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=self._build_config(system_prompt),
        )
        return response.text or ""

    async def generate_stream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        stream = self._client.aio.models.generate_content_stream(
            model=self.model,
            contents=user_prompt,
            config=self._build_config(system_prompt),
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text
