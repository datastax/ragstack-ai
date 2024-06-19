from dataclasses import dataclass
from typing import Literal, Dict, Any, Set


@dataclass(frozen=True)
class _LinkTag:
    kind: str
    tag: str
    direction: Literal["incoming", "outgoing", "bidir"]


@dataclass(frozen=True)
class LinkTag(_LinkTag):
    def __init__(self, kind: str, tag: str, direction: str) -> None:
        if self.__class__ == LinkTag:
            raise TypeError("Abstract class LinkTag cannot be instantiated")
        super().__init__(kind, tag, direction)


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
