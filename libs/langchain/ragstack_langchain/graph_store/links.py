from dataclasses import dataclass
from typing import Literal, Dict, Any, Set, Union

from langchain_core.documents import Document


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


LINKS = "links"


def get_links(doc_or_md: Union[Document, Dict[str, Any]]) -> Set[Link]:
    """Get the links from a document or metadata.
    Args:
        doc_or_md: The metadata to get the link tags from.
    Returns:
        The set of link tags from the document or metadata.
    """

    if isinstance(doc_or_md, Document):
        doc_or_md = doc_or_md.metadata

    links = doc_or_md.setdefault(LINKS, set())
    if not isinstance(links, Set):
        links = set(links)
        doc_or_md[LINKS] = links
    return links


def add_links(doc_or_md: Union[Document, Dict[str, Any]], *links: Link) -> None:
    """Add links to the given metadata.
    Args:
        doc_or_md: The document or metadata to add the links to.
        *links: The links to add to the metadata.
    """
    get_links(doc_or_md).update(links)
