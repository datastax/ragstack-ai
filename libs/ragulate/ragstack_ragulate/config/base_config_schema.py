from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .objects import Config


class BaseConfigSchema(ABC):
    """Base config schema."""

    @abstractmethod
    def version(self) -> float:
        """Returns the config file version."""

    @abstractmethod
    def schema(self) -> dict[str, Any]:
        """Returns the config file schema."""

    @abstractmethod
    def parse_document(self, document: dict[str, Any]) -> Config:
        """Parses a validated config file and returns a Config object."""
