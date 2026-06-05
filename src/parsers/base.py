from abc import ABC, abstractmethod
from src.models import ParsedConversation


class AbstractParser(ABC):

    @abstractmethod
    def can_parse(self, raw: dict) -> bool:
        ...

    @abstractmethod
    def parse(self, raw: dict) -> ParsedConversation:
        ...
