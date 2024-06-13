from dataclasses import dataclass
from typing import Literal, Dict, Any, Set


@dataclass(frozen=True)
class Link:
    kind: str
    direction: Literal["incoming", "outgoing", "bidir"]

    def __post_init__(self):
        if self.__class__ == LinkTag:
            raise TypeError("Abstract class Link cannot be instantiated")


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


LINK_SET = "link_set"


def get_link_set(doc_or_md: Dict[str, Any]) -> Set[Link]:
    """Get the link set from a document or metadata.

    Args:
        doc_or_md: The document or metadata to get the link tags from.

    Returns:
        The set of link tags from the document or metadata.
    """
    link_set = doc_or_md.setdefault(LINK_SET, set())
    if not isinstance(link_set, Set):
        link_set = set(link_set)
        doc_or_md[LINK_SET] = link_set
    return link_set
