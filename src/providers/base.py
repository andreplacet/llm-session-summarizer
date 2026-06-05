from abc import ABC, abstractmethod
from typing import AsyncIterator


class AbstractProvider(ABC):

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...

    @abstractmethod
    async def generate_stream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        ...
