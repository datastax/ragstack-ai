from __future__ import annotations

import abc
import dataclasses
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Generic, Iterable, Iterator, Literal, Set, Sequence, TypeVar, Union

import asyncstdlib
from langchain_core.runnables import run_in_executor
from langchain_core.documents import Document, BaseDocumentTransformer
from pydantic import BaseModel
from ._utils import strict_zip


class LinkTag(BaseModel, abc.ABC):
    kind: str
    direction: Literal["incoming", "outgoing", "bidir"]
    tag: str

    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))

class OutgoingLinkTag(LinkTag):
    direction: Literal["outgoing"] = "outgoing"

class IncomingLinkTag(LinkTag):
    direction: Literal["incoming"] = "incoming"

class BidirLinkTag(LinkTag):
    direction: Literal["bidir"] = "bidir"

LINK_TAGS = "link_tags"

def get_link_tags(doc_or_md: Union[Document, Dict[str, Any]]) -> Set[LinkTag]:
    """Get the link-tag set from a document or metadata.

    Args:
        doc_or_md: The document or metadata to get the link tags from.

    Returns:
        The set of link tags from the document or metadata.
    """
    if isinstance(doc_or_md, Document):
        doc_or_md = doc_or_md.metadata

    link_tags = doc_or_md.setdefault(LINK_TAGS, set())
    if not isinstance(link_tags, Set):
        link_tags = set(link_tags)
        doc_or_md[LINK_TAGS] = link_tags
    return link_tags

InputT = TypeVar("InputT")
class EdgeExtractor(ABC, Generic[InputT]):
    @abstractmethod
    def extract_one(self, document: Document, input: InputT):
        """Add edges from each `input` to the corresponding documents.

        Args:
            document: Document to add the link tags to.
            inputs: The input content to extract edges from.
        """

    def extract(self, documents: Iterable[Document], inputs: Iterable[InputT]) -> Iterator[Set[LinkTag]]:
        """Add edges from each `input` to the corresponding documents.

        Args:
            documents: The documents to add the link tags to.
            inputs: The input content to extract edges from.
        """
        for (document, input) in strict_zip(documents, inputs):
            self.extract_one(document, input)