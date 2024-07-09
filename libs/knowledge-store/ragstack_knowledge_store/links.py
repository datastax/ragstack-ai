from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Link:
    """A link to a tag in the graph."""

    kind: str
    direction: Literal["in", "out", "bidir"]
    tag: str

    @staticmethod
    def incoming(kind: str, tag: str) -> "Link":
        """Create an incoming link."""
        return Link(kind=kind, direction="in", tag=tag)

    @staticmethod
    def outgoing(kind: str, tag: str) -> "Link":
        """Create an outgoing link."""
        return Link(kind=kind, direction="out", tag=tag)

    @staticmethod
    def bidir(kind: str, tag: str) -> "Link":
        """Create a bidirectional link."""
        return Link(kind=kind, direction="bidir", tag=tag)
