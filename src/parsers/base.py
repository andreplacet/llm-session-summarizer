from abc import ABC, abstractmethod
from typing import Union
from src.models import ParsedConversation


class AbstractParser(ABC):

    @abstractmethod
    def can_parse(self, raw: Union[dict, str]) -> bool:
        ...

    @abstractmethod
    def parse(self, raw: Union[dict, str]) -> ParsedConversation:
        ...
