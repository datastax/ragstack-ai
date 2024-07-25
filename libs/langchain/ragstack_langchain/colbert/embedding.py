from typing import List, Optional

from langchain_core.embeddings import Embeddings
from ragstack_colbert import DEFAULT_COLBERT_MODEL, ColbertEmbeddingModel
from ragstack_colbert.base_embedding_model import BaseEmbeddingModel
from typing_extensions import override


class TokensEmbeddings(Embeddings):
    """Adapter for token-based embedding models and the LangChain Embeddings."""

    def __init__(self, embedding: BaseEmbeddingModel = None):
        self.embedding = embedding or ColbertEmbeddingModel()

    @override
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    @override
    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError

    @override
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    @override
    async def aembed_query(self, text: str) -> List[float]:
        raise NotImplementedError

    def get_embedding_model(self) -> BaseEmbeddingModel:
        """Get the embedding model."""
        return self.embedding

    @staticmethod
    def colbert(
        checkpoint: str = DEFAULT_COLBERT_MODEL,
        doc_maxlen: int = 256,
        nbits: int = 2,
        kmeans_niters: int = 4,
        nranks: int = -1,
        query_maxlen: Optional[int] = None,
        verbose: int = 3,
        chunk_batch_size: int = 640,
    ):
        """Create a new ColBERT embedding model."""
        return TokensEmbeddings(
            ColbertEmbeddingModel(
                checkpoint,
                doc_maxlen,
                nbits,
                kmeans_niters,
                nranks,
                query_maxlen,
                verbose,
                chunk_batch_size,
            )
        )
