"""
This module integrates the ColBERT model with token embedding functionalities, offering tools for efficiently
encoding queries and text chunks into dense vector representations. It facilitates semantic search and
retrieval by providing optimized methods for embedding generation and manipulation.

The core component, ColbertEmbeddingModel, leverages pre-trained ColBERT models to produce embeddings suitable
for high-relevancy retrieval tasks, with support for both CPU and GPU computing environments.
"""

import logging
from typing import List, Optional

from colbert.infra import ColBERTConfig

from .base_embedding_model import BaseEmbeddingModel
from .constant import DEFAULT_COLBERT_MODEL
from .objects import Chunk, Embedding
from .text_encoder import TextEncoder


class ColbertEmbeddingModel(BaseEmbeddingModel):
    """
    A class for generating token embeddings using a ColBERT model. This class provides functionalities for
    encoding queries and document chunks into dense vector representations, facilitating semantic search and
    retrieval tasks. It leverages a pre-trained ColBERT model and supports distributed computing environments.

    The class supports both GPU and CPU operations, with GPU usage recommended for performance efficiency.
    """

    _query_maxlen: int
    _chunk_batch_size: int

    def __init__(
        self,
        checkpoint: Optional[str] = DEFAULT_COLBERT_MODEL,
        doc_maxlen: Optional[int] = 256,
        nbits: Optional[int] = 2,
        kmeans_niters: Optional[int] = 4,
        nranks: Optional[int] = -1,
        query_maxlen: Optional[int] = None,
        verbose: Optional[int] = 3,  # 3 is the default on ColBERT checkpoint
        chunk_batch_size: Optional[int] = 640,
        **kwargs,
    ):
        """
        Initializes a new instance of the ColbertEmbeddingModel class, setting up the model configuration,
        loading the necessary checkpoints, and preparing the tokenizer and encoder.

        Parameters:
            checkpoint (Optional[str]): Path or URL to the Colbert model checkpoint. Default is a pre-defined model.
            doc_maxlen (Optional[int]): Maximum number of tokens for document chunks. Should equal the chunk_size.
            nbits (Optional[int]): The number bits that each dimension encodes to.
            kmeans_niters (Optional[int]): Number of iterations for k-means clustering during quantization.
            nranks (Optional[int]): Number of ranks (processors) to use for distributed computing; -1 uses all available CPUs/GPUs.
            query_maxlen (Optional[int]): Maximum length of query tokens for embedding.
            verbose (Optional[int]): Verbosity level for logging.
            chunk_batch_size (Optional[int]): The number of chunks to batch during embedding. Defaults to 640.
            **kwargs: Additional keyword arguments for future extensions.
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

    # implements the Abstract Class Method
    def embed_texts(self, texts: List[str]) -> List[Embedding]:
        """
        Embeds a list of texts into their corresponding vector embedding representations.

        Parameters:
            texts (List[str]): A list of string texts.

        Returns:
            List[Embedding]: A list of embeddings, in the order of the input list
        """

        chunks = [
            Chunk(doc_id="dummy", chunk_id=i, text=t) for i, t in enumerate(texts)
        ]

        embedded_chunks = []

        for i in range(0, len(chunks), self._chunk_batch_size):
            chunk_batch = chunks[i : i + self._chunk_batch_size]
            embedded_chunks.extend(self._encoder.encode_chunks(chunks=chunk_batch))

        sorted_embedded_chunks = sorted(embedded_chunks, key=lambda c: c.chunk_id)

        return [c.embedding for c in sorted_embedded_chunks]

    # implements the Abstract Class Method
    def embed_query(
        self,
        query: str,
        full_length_search: Optional[bool] = False,
        query_maxlen: Optional[int] = None,
    ) -> Embedding:
        """
        Embeds a single query text into its vector representation.

        If the query has fewer than query_maxlen tokens it will be padded with BERT special [mast] tokens.

        Parameters:
            query (str): The query string to encode.
            full_length_search (Optional[bool]): Indicates whether to encode the query for a full-length search.
                                                  Defaults to False.
            query_maxlen (int): The fixed length for the query token embedding. If None, uses a dynamically calculated value.

        Returns:
            Embedding: A vector embedding representation of the query text
        """

        if query_maxlen is None:
            query_maxlen = -1

        query_maxlen = max(query_maxlen, self._query_maxlen)
        return self._encoder.encode_query(
            text=query, query_maxlen=query_maxlen, full_length_search=full_length_search
        )
