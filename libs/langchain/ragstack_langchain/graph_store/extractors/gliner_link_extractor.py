from typing import Any, Dict, Iterable, List, Optional, Set

from ragstack_langchain.graph_store.extractors.link_extractor import LinkExtractor
from ragstack_langchain.graph_store.links import Link

# TypeAlias is not available in Python 2.9, we can't use that or the newer `type`.
GLiNERInput = str


class GLiNERLinkExtractor(LinkExtractor[GLiNERInput]):
    def __init__(
        self,
        labels: List[str],
        *,
        kind: str = "entity",
        model: str = "urchade/gliner_mediumv2.1",
        extract_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Extract keywords using GLiNER.

        Args:
            kind: Kind of links to produce with this extractor.
            labels: List of kinds of entities to extract.
            model: GLiNER model to use.
            extract_kwargs: Keyword arguments to pass to GLiNER.
        """
        try:
            from gliner import GLiNER

            self._model = GLiNER.from_pretrained(model)

        except ImportError:
            raise ImportError(
                "gliner is required for GLiNERLinkExtractor. "
                "Please install it with `pip install gliner`."
            ) from None

        self._labels = labels
        self._kind = kind
        self._extract_kwargs = extract_kwargs or {}

    def extract_one(self, input: GLiNERInput) -> Set[Link]:
        return next(self.extract_many([input]))

    def extract_many(
        self,
        inputs: Iterable[GLiNERInput],
    ) -> Iterable[Set[Link]]:
        strs = [i if isinstance(i, str) else i.page_content for i in inputs]
        for entities in self._model.batch_predict_entities(
            strs, self._labels, **self._extract_kwargs
        ):
            yield {
                Link.bidir(kind=f"{self._kind}:{e['label']}", tag=e["text"])
                for e in entities
            }
