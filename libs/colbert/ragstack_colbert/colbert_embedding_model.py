"""ColBERT Embedding Model.

This module integrates the ColBERT model with token embedding functionalities,
offering tools for efficiently encoding queries and text chunks into dense vector
representations. It facilitates semantic search and retrieval by providing optimized
methods for embedding generation and manipulation.

The core component, ColbertEmbeddingModel, leverages pre-trained ColBERT models to
produce embeddings suitable for high-relevancy retrieval tasks,
with support for both CPU and GPU computing environments.
"""

from __future__ import annotations

from colbert.infra import ColBERTConfig
from typing_extensions import override

from .base_embedding_model import BaseEmbeddingModel
from .constant import DEFAULT_COLBERT_MODEL
from .objects import Chunk, Embedding
from .text_encoder import TextEncoder


class ColbertEmbeddingModel(BaseEmbeddingModel):
    """ColBERT embedding model.

    A class for generating token embeddings using a ColBERT model. This class
    provides functionalities for encoding queries and document chunks into dense vector
    representations, facilitating semantic search and retrieval tasks. It leverages a
    pre-trained ColBERT model and supports distributed computing environments.

    The class supports both GPU and CPU operations, with GPU usage recommended for
    performance efficiency.
    """

    _query_maxlen: int
    _chunk_batch_size: int

    def __init__(
        self,
        checkpoint: str = DEFAULT_COLBERT_MODEL,
        doc_maxlen: int = 256,
        nbits: int = 2,
        kmeans_niters: int = 4,
        nranks: int = -1,
        query_maxlen: int | None = None,
        verbose: int = 3,  # 3 is the default on ColBERT checkpoint
        chunk_batch_size: int = 640,
    ):
        """Initializes a new instance of the ColbertEmbeddingModel class.

        Initializes a new instance of the ColbertEmbeddingModel class setting up the
        model configuration, loading the necessary checkpoints, and preparing the
        tokenizer and encoder.

        Args:
            checkpoint: Path or URL to the Colbert model checkpoint.
                Default is a pre-defined model.
            doc_maxlen: Maximum number of tokens for document chunks.
                Should equal the chunk_size.
            nbits: The number bits that each dimension encodes to.
            kmeans_niters: Number of iterations for k-means clustering
                during quantization.
            nranks: Number of ranks (processors) to use for distributed
                computing; -1 uses all available CPUs/GPUs.
            query_maxlen: Maximum length of query tokens for embedding.
            verbose: Verbosity level for logging.
            chunk_batch_size: The number of chunks to batch during
                embedding. Defaults to 640.
        """
        if query_maxlen is None:
            query_maxlen = -1

        self._query_maxlen = query_maxlen
        self._chunk_batch_size = chunk_batch_size

        colbert_config = ColBERTConfig(
            doc_maxlen=doc_maxlen,
            nbits=nbits,
            kmeans_niters=kmeans_niters,
            nranks=nranks,
            checkpoint=checkpoint,
        )
        self._encoder = TextEncoder(config=colbert_config, verbose=verbose)

    @override
    def embed_texts(self, texts: list[str]) -> list[Embedding]:
        chunks = [
            Chunk(doc_id="dummy", chunk_id=i, text=t) for i, t in enumerate(texts)
        ]

        embedded_chunks = []

        for i in range(0, len(chunks), self._chunk_batch_size):
            chunk_batch = chunks[i : i + self._chunk_batch_size]
            embedded_chunks.extend(self._encoder.encode_chunks(chunks=chunk_batch))

        sorted_embedded_chunks = sorted(embedded_chunks, key=lambda c: c.chunk_id)

        return [c.embedding or [] for c in sorted_embedded_chunks]

    @override
    def embed_query(
        self,
        query: str,
        full_length_search: bool = False,
        query_maxlen: int | None = None,
    ) -> Embedding:
        if query_maxlen is None:
            query_maxlen = -1

        query_maxlen = max(query_maxlen, self._query_maxlen)
        return self._encoder.encode_query(
            text=query, query_maxlen=query_maxlen, full_length_search=full_length_search
        )
