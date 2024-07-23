"""ColBERT Retriever Module.

This module integrates text embedding retrieval and similarity computation
functionalities with a vector store backend, optimized for high-performance operations
in large-scale text retrieval applications.

Note:
The implementation assumes the availability of a GPU for optimal performance but is
designed to fallback to CPU computation if needed.
This flexibility ensures that the retrieval system can be deployed in a variety of
hardware environments.
"""

import asyncio
import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from typing_extensions import override

from .base_database import BaseDatabase
from .base_embedding_model import BaseEmbeddingModel
from .base_retriever import BaseRetriever
from .objects import Chunk, Embedding, Vector


def all_gpus_support_fp16(is_cuda: bool = False) -> bool:
    """Check if all available GPU devices support FP16 (half-precision) operations.

    Returns:
        True if all GPUs support FP16, False otherwise.
    """
    if not is_cuda:
        return False

    for device_id in range(torch.cuda.device_count()):
        compute_capability = torch.cuda.get_device_capability(device_id)
        # FP16 support requires compute capability of 5.3 or higher
        min_compute_capability_major = 5
        min_compute_capability_minor = 3
        if compute_capability[0] < min_compute_capability_major or (
            compute_capability[0] == min_compute_capability_major
            and compute_capability[1] < min_compute_capability_minor
        ):
            logging.info(
                "Device %s with compute capability %s does not support FP16 "
                "(half-precision) operations. Using FP32 (full-precision) operations.",
                device_id,
                compute_capability,
            )
            return False

    # If all GPUs passed the check
    return True


def max_similarity_torch(
    query_vector: Vector,
    chunk_embedding: Embedding,
    is_cuda: bool = False,
    is_fp16: bool = False,
) -> float:
    """Calculates the maximum similarity for a query vector and a chunk embedding.

    Calculates the maximum similarity (dot product) between a query vector and a
    chunk embedding, leveraging PyTorch for efficient computation.

    Args:
        query_vector: A list of float representing the query text.
        chunk_embedding: A list of Vector, each representing an chunk
            embedding vector.
        is_cuda: A flag indicating whether to use CUDA (GPU)
            for computation. Defaults to False.
        is_fp16: A flag indicating whether to half-precision floating point
            operations on CUDA (GPU).
            Has no effect on CPU computation. Defaults to False.

    Returns:
        A tensor containing the highest similarity score (dot product value)
            found between the query vector and any of the embedding vectors in the list.

    Note:
        This function is designed to run on GPU for enhanced performance but can also
        execute on CPU.
    """
    # Convert inputs to tensors
    query_tensor = torch.Tensor(query_vector)
    embedding_tensor = torch.stack([torch.Tensor(v) for v in chunk_embedding])

    if is_cuda:
        device = torch.device("cuda")
        query_tensor = query_tensor.to(device)
        embedding_tensor = embedding_tensor.to(device)

        # Use half-precision operations if supported
        if is_fp16:
            query_tensor = query_tensor.half()
            embedding_tensor = embedding_tensor.half()

    # Perform the dot product operation
    sims = torch.matmul(embedding_tensor, query_tensor)

    # Find the maximum similarity
    max_sim = torch.max(sims)

    # returns a tensor; the item() is the score
    return float(max_sim.item())


class ColbertRetriever(BaseRetriever):
    """ColBERT Retriever.

    A retriever class that implements the retrieval of text chunks from a vector
    store, based on their semantic similarity to a given query.
    This implementation leverages the ColBERT model for generating embeddings of
    the query.

    Args:
        database (BaseDatabase): The data store to be used for retrieving
            embeddings.
        embedding_model (BaseEmbeddingModel): The ColBERT embeddings model to be
            used for encoding queries.

    Note:
        The class is designed to work with a GPU for optimal performance but will
        automatically fall back to CPU computation if a GPU is not available.
    """

    _database: BaseDatabase
    _embedding_model: BaseEmbeddingModel
    _is_cuda: bool
    _is_fp16: bool

    class Config:
        """Pydantic configuration for the ColbertRetriever class."""

        arbitrary_types_allowed = True

    def __init__(
        self,
        database: BaseDatabase,
        embedding_model: BaseEmbeddingModel,
    ):
        self._database = database
        self._embedding_model = embedding_model
        self._is_cuda = torch.cuda.is_available()
        self._is_fp16 = all_gpus_support_fp16(self._is_cuda)

    async def _query_relevant_chunks(
        self, query_embedding: Embedding, top_k: int
    ) -> Set[Chunk]:
        """Queries for the top_k most relevant chunks for each query token."""
        chunks: Set[Chunk] = set()
        # Collect all tasks
        tasks = [
            self._database.search_relevant_chunks(vector=v, n=top_k)
            for v in query_embedding
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle potential exceptions
        for result in results:
            if isinstance(result, BaseException):
                logging.error(
                    "Issue on database.get_relevant_chunks()",
                    exc_info=result,
                )
            else:
                chunks.update(result)

        return chunks

    async def _get_chunk_embeddings(self, chunks: Set[Chunk]) -> List[Chunk]:
        """Retrieves Chunks with `doc_id`, `chunk_id`, and `embedding` set."""
        # Collect all tasks
        tasks = [
            self._database.get_chunk_embedding(doc_id=c.doc_id, chunk_id=c.chunk_id)
            for c in chunks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle potential exceptions
        chunk_embeddings = []
        for result in results:
            if isinstance(result, BaseException):
                logging.error(
                    "Issue on database.get_chunk_embeddings()",
                    exc_info=result,
                )
            else:
                chunk_embeddings.append(result)

        return chunk_embeddings

    def _score_chunks(
        self, query_embedding: Embedding, chunk_embeddings: List[Chunk]
    ) -> Dict[Chunk, float]:
        """Process the retrieved chunk data to calculate scores."""
        chunk_scores = {}
        for chunk in chunk_embeddings:
            if not chunk.embedding:
                continue
            chunk_scores[chunk] = sum(
                max_similarity_torch(
                    query_vector=query_vector,
                    chunk_embedding=chunk.embedding,
                    is_cuda=self._is_cuda,
                    is_fp16=self._is_fp16,
                )
                for query_vector in query_embedding
            )
        return chunk_scores

    async def _get_chunk_data(
        self,
        chunks: List[Chunk],
        include_embedding: bool = False,
    ) -> List[Chunk]:
        """Fetches text and metadata for each chunk.

        Returns:
            List[Chunk]: A list of chunks with `doc_id`, `chunk_id`, `text`, `metadata`,
                and optionally `embedding` set.
        """
        # Collect all tasks
        tasks = [
            self._database.get_chunk_data(
                doc_id=c.doc_id,
                chunk_id=c.chunk_id,
                include_embedding=include_embedding,
            )
            for c in chunks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        chunks = []
        for result in results:
            if isinstance(result, BaseException):
                logging.error(
                    "Issue on database.get_chunk_data()",
                    exc_info=result,
                )
            else:
                chunks.append(result)

        return chunks

    @override
    async def atext_search(
        self,
        query_text: str,
        k: Optional[int] = 5,
        query_maxlen: Optional[int] = None,
        include_embedding: bool = False,
        **kwargs: Any,
    ) -> List[Tuple[Chunk, float]]:
        query_embedding = self._embedding_model.embed_query(
            query=query_text, query_maxlen=query_maxlen
        )

        return await self.aembedding_search(
            query_embedding=query_embedding,
            k=k,
            include_embedding=include_embedding,
            **kwargs,
        )

    @override
    async def aembedding_search(
        self,
        query_embedding: Embedding,
        k: Optional[int] = 5,
        include_embedding: bool = False,
        **kwargs: Any,
    ) -> List[Tuple[Chunk, float]]:
        if k is None:
            k = 5
        top_k = max(math.floor(len(query_embedding) / 2), 16)
        logging.debug(
            "based on query length of %s tokens, retrieving %s results per "
            "token-embedding",
            len(query_embedding),
            top_k,
        )

        # search for relevant chunks (only with `doc_id` and `chunk_id` set)
        relevant_chunks: Set[Chunk] = await self._query_relevant_chunks(
            query_embedding=query_embedding, top_k=top_k
        )

        # get the embedding for each chunk
        # (with `doc_id`, `chunk_id`, and `embedding` set)
        chunk_embeddings: List[Chunk] = await self._get_chunk_embeddings(
            chunks=relevant_chunks
        )

        # score the chunks using max_similarity
        chunk_scores: Dict[Chunk, float] = self._score_chunks(
            query_embedding=query_embedding,
            chunk_embeddings=chunk_embeddings,
        )

        # only keep the top k sorted results
        top_k_chunks: List[Chunk] = sorted(
            chunk_scores, key=lambda c: chunk_scores.get(c, 0), reverse=True
        )[:k]

        chunks: List[Chunk] = await self._get_chunk_data(
            chunks=top_k_chunks, include_embedding=include_embedding
        )

        return [(chunk, chunk_scores[chunk]) for chunk in chunks]

    @override
    def text_search(
        self,
        query_text: str,
        k: Optional[int] = 5,
        query_maxlen: Optional[int] = None,
        include_embedding: bool = False,
        **kwargs: Any,
    ) -> List[Tuple[Chunk, float]]:
        return asyncio.run(
            self.atext_search(
                query_text=query_text,
                k=k,
                query_maxlen=query_maxlen,
                include_embedding=include_embedding,
                **kwargs,
            )
        )

    @override
    def embedding_search(
        self,
        query_embedding: Embedding,
        k: Optional[int] = 5,
        include_embedding: bool = False,
        **kwargs: Any,
    ) -> List[Tuple[Chunk, float]]:
        return asyncio.run(
            self.aembedding_search(
                query_embedding=query_embedding,
                k=k,
                include_embedding=include_embedding,
            )
        )
