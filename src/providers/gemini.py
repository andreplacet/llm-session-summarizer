import os
from typing import AsyncIterator, Optional

from google import genai

from src.providers.base import AbstractProvider

AVAILABLE_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-preview",
]

MODEL_ALIASES = {
    "flash": "gemini-3-flash-preview",
    "pro": "gemini-3.1-pro-preview",
    "flash-3.1": "gemini-3.1-flash-preview",
}

DEFAULT_MODEL = "gemini-3-flash-preview"


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
            raise ValueError(
                "GEMINI_API_KEY não encontrada. Configure no .env ou passe explicitamente."
            )
        self._client = genai.Client(api_key=api_key)

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        contents = self._build_contents(system_prompt, user_prompt)
        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=genai.types.GenerateContentConfig(temperature=self.temperature),
        )
        return response.text or ""

    async def generate_stream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        contents = self._build_contents(system_prompt, user_prompt)
        stream = self._client.aio.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=genai.types.GenerateContentConfig(temperature=self.temperature),
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text

    def _build_contents(self, system_prompt: str, user_prompt: str) -> str:
        return f"{system_prompt}\n\n---\n\n{user_prompt}"
