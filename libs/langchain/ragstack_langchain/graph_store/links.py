from typing import Dict, Any, Iterable, Set, Union

from langchain_core.documents import Document
from ragstack_knowledge_store.links import Link


METADATA_LINKS_KEY = "links"


def get_links(doc: Document) -> Set[Link]:
    """Get the links from a document.
    Args:
        doc: The document to get the link tags from.
    Returns:
        The set of link tags from the document.
    """

    links = doc.metadata.setdefault(METADATA_LINKS_KEY, set())
    if not isinstance(links, Set):
        # Convert to a set and remember that.
        links = set(links)
        doc.metadata[METADATA_LINKS_KEY] = links
    return links


def add_links(doc: Document, *links: Link) -> None:
    """Add links to the given metadata.
    Args:
        doc: The document to add the links to.
        *links: The links to add to the document.
    """
    get_links(doc).update(links)