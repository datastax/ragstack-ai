from abc import ABC, abstractmethod
from typing import Any, Dict

from .objects import Config


class BaseConfigSchema(ABC):

    @abstractmethod
    def version() -> float:
        """returns the config file version"""

    @abstractmethod
    def schema(self) -> Dict[str, Any]:
        """returns the config file schema"""

    @abstractmethod
    def parse_document(self, document: Dict[str, Any]) -> Config:
        """parses a validated config file and returns a Config object"""
