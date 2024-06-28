from typing import Callable, Iterable, Set, TypeVar

from ragstack_langchain.graph_store.extractors.link_extractor import LinkExtractor
from ragstack_langchain.graph_store.links import Link

InputT = TypeVar("InputT")
UnderlyingInputT = TypeVar("UnderlyingInputT")


class LinkExtractorAdapter(LinkExtractor[InputT]):
    def __init__(
        self,
        underlying: LinkExtractor[UnderlyingInputT],
        transform: Callable[[InputT], UnderlyingInputT],
    ) -> None:
        self._underlying = underlying
        self._transform = transform

    def extract_one(self, input: InputT) -> Set[Link]:
        return self.extract_one(self._transform(input))

    def extract_many(self, inputs: Iterable[InputT]) -> Iterable[Set[Link]]:
        underlying_inputs = [self._transform(input) for input in inputs]
        return self._underlying.extract_many(underlying_inputs)
