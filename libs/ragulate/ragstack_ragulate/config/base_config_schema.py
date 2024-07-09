from abc import ABC, abstractmethod
from typing import Any, Dict

from .objects import Config


class BaseConfigSchema(ABC):
    """Base config schema."""

    @abstractmethod
    def version(self) -> float:
        """Returns the config file version."""

    @abstractmethod
    def schema(self) -> Dict[str, Any]:
        """Returns the config file schema."""

    @abstractmethod
    def parse_document(self, document: Dict[str, Any]) -> Config:
        """Parses a validated config file and returns a Config object."""
