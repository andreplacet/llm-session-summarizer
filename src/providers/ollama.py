from __future__ import annotations

import json
from typing import AsyncIterator, Optional

import httpx

from src.providers.base import AbstractProvider

OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaProvider(AbstractProvider):

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = 0.3,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        prompt = self._build_prompt(system_prompt, user_prompt)
        async with httpx.AsyncClient(timeout=600) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")

    async def generate_stream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        prompt = self._build_prompt(system_prompt, user_prompt)
        async with httpx.AsyncClient(timeout=600) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {"temperature": self.temperature},
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = chunk.get("response", "")
                    if text:
                        yield text
                    if chunk.get("done"):
                        break

    def _build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        return f"{system_prompt}\n\n---\n\n{user_prompt}"

    @staticmethod
    async def list_models(base_url: str = OLLAMA_BASE_URL) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{base_url.rstrip('/')}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
