from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Link:
    kind: str
    direction: Literal["incoming", "outgoing", "bidir"]
    tag: str

    @staticmethod
    def incoming(kind: str, tag: str) -> "Link":
        return Link(kind=kind, direction="incoming", tag=tag)

    @staticmethod
    def outgoing(kind: str, tag: str) -> "Link":
        return Link(kind=kind, direction="outgoing", tag=tag)

    @staticmethod
    def bidir(kind: str, tag: str) -> "Link":
        return Link(kind=kind, direction="bidir", tag=tag)