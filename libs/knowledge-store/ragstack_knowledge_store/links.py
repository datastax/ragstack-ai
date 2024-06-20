from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Link:
    kind: str
    direction: Literal["incoming", "outgoing", "bidir"]

    def __post_init__(self):
        if self.__class__ in [Link, LinkTag]:
            raise TypeError(
                f"Abstract class {self.__class__.__name__} cannot be instantiated"
            )


@dataclass(frozen=True)
class LinkTag(Link):
    tag: str


@dataclass(frozen=True)
class OutgoingLinkTag(LinkTag):
    def __init__(self, kind: str, tag: str) -> None:
        super().__init__(kind=kind, tag=tag, direction="outgoing")


@dataclass(frozen=True)
class IncomingLinkTag(LinkTag):
    def __init__(self, kind: str, tag: str) -> None:
        super().__init__(kind=kind, tag=tag, direction="incoming")


@dataclass(frozen=True)
class BidirLinkTag(LinkTag):
    def __init__(self, kind: str, tag: str) -> None:
        super().__init__(kind=kind, tag=tag, direction="bidir")
