from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Iterable, Iterator, Set, TypeVar

from langchain_core.documents import Document
from ragstack_knowledge_store._utils import strict_zip
from ragstack_knowledge_store.link_tag import LinkTag

InputT = TypeVar("InputT")


class EdgeExtractor(ABC, Generic[InputT]):
    @abstractmethod
    def extract_one(self, document: Document, input: InputT):
        """Add edges from each `input` to the corresponding documents.

        Args:
            document: Document to add the link tags to.
            inputs: The input content to extract edges from.
        """

    def extract(
        self, documents: Iterable[Document], inputs: Iterable[InputT]
    ) -> Iterator[Set[LinkTag]]:
        """Add edges from each `input` to the corresponding documents.

        Args:
            documents: The documents to add the link tags to.
            inputs: The input content to extract edges from.
        """
        for document, input in strict_zip(documents, inputs):
            self.extract_one(document, input)
