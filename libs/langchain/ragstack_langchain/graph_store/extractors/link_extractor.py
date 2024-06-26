from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import repeat
from typing import Any, Dict, Generic, Iterable, Optional, Set, TypeVar

from ragstack_knowledge_store._utils import strict_zip

from ragstack_langchain.graph_store.links import Link

InputT = TypeVar("InputT")

METADATA_LINKS_KEY = "links"



class LinkExtractor(ABC, Generic[InputT]):
    """Interface for extracting links (incoming, outgoing, bidirectional)."""

    @abstractmethod
    def extract_one(self, input: InputT, **kwargs: Any) -> Set[Link]:
        """Add edges from each `input` to the corresponding documents.

        Args:
            input: The input content to extract edges from.
            **kwargs: Additional keyword arguments for the extractor.

        Returns:
            Set of links extracted from the input.
        """

    def extract_many(self,
                      inputs: Iterable[InputT],
                      batch_kwargs: Optional[Iterable[Dict[str, Any]]] = None,
                      **kwargs: Any):
        """Add edges from each `input` to the corresponding documents.

        Args:
            inputs: The input content to extract edges from.
            batch_kwargs: Iterable of keyword arguments for each input.
                Defaults to empty dictionaries.
            **kwargs: Additional arguments to the extractor.

        Returns:
            Iterable over the set of links extracted from the input.
        """
        for (input, kwargs) in zip(inputs, batch_kwargs or repeat({})):
            yield self.extract_one(input, **kwargs)
