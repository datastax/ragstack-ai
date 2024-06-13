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


LINK_TAGS = "link_tags"


def get_link_tags(doc_or_md: Dict[str, Any]) -> Set[LinkTag]:
    """Get the link-tag set from a document or metadata.

    Args:
        doc_or_md: The document or metadata to get the link tags from.

    Returns:
        The set of link tags from the document or metadata.
    """
    link_tags = doc_or_md.setdefault(LINK_TAGS, set())
    if not isinstance(link_tags, Set):
        link_tags = set(link_tags)
        doc_or_md[LINK_TAGS] = link_tags
    return link_tags
