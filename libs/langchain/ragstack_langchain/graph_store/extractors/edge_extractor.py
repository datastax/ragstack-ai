from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Iterable, Set, TypeVar

from langchain_core.documents import Document
from ragstack_knowledge_store._utils import strict_zip

from ragstack_langchain.graph_store.links import Link

InputT = TypeVar("InputT")

METADATA_LINKS_KEY = "links"


class EdgeExtractor(ABC, Generic[InputT]):
    @abstractmethod
    def extract_one(self, document: Document, input: InputT) -> None:
        """Add edges from each `input` to the corresponding documents.

        Args:
            document: Document to add the link tags to.
            input: The input content to extract edges from.
        """

    def extract(self, documents: Iterable[Document], inputs: Iterable[InputT]) -> None:
        """Add edges from each `input` to the corresponding documents.

        Args:
            documents: The documents to add the link tags to.
            inputs: The input content to extract edges from.
        """
        for document, input in strict_zip(documents, inputs):
            self.extract_one(document, input)
