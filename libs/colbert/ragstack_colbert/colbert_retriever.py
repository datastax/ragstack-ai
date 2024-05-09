"""
This module integrates text embedding retrieval and similarity computation functionalities with a vector
store backend, optimized for high-performance operations in large-scale text retrieval applications.

Note:
The implementation assumes the availability of a GPU for optimal performance but is designed to fallback
to CPU computation if needed. This flexibility ensures that the retrieval system can be deployed in a
variety of hardware environments.
"""

import asyncio
import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from .base_database import BaseDatabase
from .base_embedding_model import BaseEmbeddingModel
from .base_retriever import BaseRetriever
from .objects import Chunk, Embedding, Vector


def all_gpus_support_fp16(is_cuda: Optional[bool] = False):
    """
    Check if all available GPU devices support FP16 (half-precision) operations.

    Returns:
        bool: True if all GPUs support FP16, False otherwise.
    """
    if not is_cuda:
        return False

    for device_id in range(torch.cuda.device_count()):
        compute_capability = torch.cuda.get_device_capability(device_id)
        # FP16 support requires compute capability of 5.3 or higher
        if compute_capability[0] < 5 or (
            compute_capability[0] == 5 and compute_capability[1] < 3
        ):
            logging.info(
                f"Device {device_id} with compute capability {compute_capability} does not support FP16 (half-precision) operations. Using FP32 (full-precision) operations."
            )
            return False

    # If all GPUs passed the check
    return True


def max_similarity_torch(
    query_vector: Vector,
    chunk_embedding: Embedding,
    is_cuda: Optional[bool] = False,
    is_fp16: Optional[bool] = False,
) -> float:
    """
    Calculates the maximum similarity (dot product) between a query vector and a chunk embedding,
    leveraging PyTorch for efficient computation.

    Parameters:
        query_vector (Vector): A list of float representing the query text.
        chunk_embedding (Embedding): A list of Vector, each representing an chunk embedding vector.
        is_cuda (Optional[bool]): A flag indicating whether to use CUDA (GPU) for computation. Defaults to False.
        is_fp16 (bool): A flag indicating whether to half-precision floating point operations on CUDA (GPU).
                        Has no effect on CPU computation. Defaults to False.

    Returns:
        Tensor: A tensor containing the highest similarity score (dot product value) found between the query vector
                and any of the embedding vectors in the list.

    Note:
        This function is designed to run on GPU for enhanced performance but can also execute on CPU.
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


def get_trace(e: Exception) -> str:
    trace = ""
    tb = e.__traceback__
    while tb is not None:
        trace += f"\tFile: {tb.tb_frame.f_code.co_filename} Line: {tb.tb_lineno}\n"
        tb = tb.tb_next
    return trace


class ColbertRetriever(BaseRetriever):
    """
    A retriever class that implements the retrieval of text chunks from a vector store, based on
    their semantic similarity to a given query. This implementation leverages the ColBERT model for
    generating embeddings of the query.

    Attributes:
        vector_store (BaseVectorStore): The vector store instance where chunks are stored.
        embedding_model (BaseEmbeddingModel): The ColBERT embeddings model for encoding queries.
        is_cuda (bool): A flag indicating whether to use CUDA (GPU) for computation.
        is_fp16 (bool): A flag indicating whether to half-precision floating point operations on CUDA (GPU).
                        Has no effect on CPU computation.

    Note:
        The class is designed to work with a GPU for optimal performance but will automatically fall back to CPU
        computation if a GPU is not available.
    """

    _database: BaseDatabase
    _embedding_model: BaseEmbeddingModel
    _is_cuda: bool
    _is_fp16: bool

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        database: BaseDatabase,
        embedding_model: BaseEmbeddingModel,
    ):
        """
        Initializes the retriever with a specific vector store and Colbert embeddings model.

        Parameters:
            database (BaseDatabase): The data store to be used for retrieving embeddings.
            embedding_model (BaseEmbeddingModel): The ColBERT embeddings model to be used for encoding
                                                         queries.
        """

        self._database = database
        self._embedding_model = embedding_model
        self._is_cuda = torch.cuda.is_available()
        self._is_fp16 = all_gpus_support_fp16(self._is_cuda)

    def close(self) -> None:
        """
        Closes any open resources held by the retriever.
        """
        pass

    async def _query_relevant_chunks(
        self, query_embedding: Embedding, top_k: int
    ) -> Set[Chunk]:
        """
        Retrieves the top_k ANN Chunks (`doc_id` and `chunk_id` only) for each embedded query token.
        """
        chunks: Set[Chunk] = set()
        # Collect all tasks
        tasks = [
            self._database.search_relevant_chunks(vector=v, n=top_k)
            for v in query_embedding
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle potential exceptions
        for result in results:
            if isinstance(result, Exception):
                logging.error(
                    f"Issue on database.get_relevant_chunks(): {result} at {get_trace(result)}"
                )
            else:
                chunks.update(result)

        return chunks

    async def _get_chunk_embeddings(self, chunks: Set[Chunk]) -> List[Chunk]:
        """
        Retrieves Chunks with `doc_id`, `chunk_id`, and `embedding` set.
        """
        # Collect all tasks
        tasks = [
            self._database.get_chunk_embedding(doc_id=c.doc_id, chunk_id=c.chunk_id)
            for c in chunks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle potential exceptions
        for result in results:
            if isinstance(result, Exception):
                logging.error(
                    f"Issue on database.get_chunk_embeddings(): {result} at {get_trace(result)}"
                )

        return results

    def _score_chunks(
        self, query_embedding: Embedding, chunk_embeddings: List[Chunk]
    ) -> Dict[Chunk, float]:
        """
        Process the retrieved chunk data to calculate scores.
        """
        chunk_scores = {}
        for chunk in chunk_embeddings:
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
        include_embedding: Optional[bool] = False,
    ) -> List[Chunk]:
        """
        Fetches text and metadata for each chunk.

        Returns:
            List[Chunk]: A list of chunks with `doc_id`, `chunk_id`, `text`, `metadata`, and optionally `embedding` set.
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

        for result in results:
            if isinstance(result, Exception):
                logging.error(
                    f"Issue on database.get_chunk_data(): {result} at {get_trace(result)}"
                )

        return results

    async def atext_search(
        self,
        query_text: str,
        k: Optional[int] = 5,
        query_maxlen: Optional[int] = None,
        include_embedding: Optional[bool] = False,
        **kwargs: Any,
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieves a list of text chunks most relevant to the given query, using semantic similarity as the criteria.

        Parameters:
            query_text (str): The query text to search for relevant text chunks.
            k (Optional[int]): The number of top results to retrieve. Default 5.
            query_maxlen (Optional[int]): The maximum length of the query to consider. If None, the
                                          maxlen will be dynamically generated.
            include_embedding (Optional[bool]): Optional (default False) flag to include the
                                                embedding vectors in the returned chunks
            **kwargs (Any): Additional parameters that implementations might require for customized
                            retrieval operations.

        Returns:
            List[Tuple[Chunk, float]]: A list of retrieved Chunk, float Tuples, each representing a text chunk that is relevant
                                  to the query, along with its similarity score.
        """

        query_embedding = self._embedding_model.embed_query(
            query=query_text, query_maxlen=query_maxlen
        )

        return await self.aembedding_search(
            query_embedding=query_embedding,
            k=k,
            include_embedding=include_embedding,
            **kwargs,
        )

    async def aembedding_search(
        self,
        query_embedding: Embedding,
        k: Optional[int] = 5,
        include_embedding: Optional[bool] = False,
        **kwargs: Any,
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieves a list of text chunks most relevant to the given query, using semantic similarity as the criteria.

        Parameters:
            query_embedding (Embedding): The query embedding to search for relevant text chunks.
            k (Optional[int]): The number of top results to retrieve. Default 5.
            include_embedding (Optional[bool]): Optional (default False) flag to include the
                                                embedding vectors in the returned chunks
            **kwargs (Any): Additional parameters that implementations might require for customized
                            retrieval operations.

        Returns:
            List[Tuple[Chunk, float]]: A list of retrieved Chunk, float Tuples, each representing a text chunk that is relevant
                                  to the query, along with its similarity score.
        """

        top_k = max(math.floor(len(query_embedding) / 2), 16)
        logging.debug(
            f"based on query length of {len(query_embedding)} tokens, retrieving {top_k} results per token-embedding"
        )

        # search for relevant chunks (only with `doc_id` and `chunk_id` set)
        relevant_chunks: List[Chunk] = await self._query_relevant_chunks(
            query_embedding=query_embedding, top_k=top_k
        )

        # get the embedding for each chunk (with `doc_id`, `chunk_id`, and `embedding` set)
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
            chunk_scores, key=chunk_scores.get, reverse=True
        )[:k]

        chunks: List[Chunk] = await self._get_chunk_data(
            chunks=top_k_chunks, include_embedding=include_embedding
        )

        return [(chunk, chunk_scores[chunk]) for chunk in chunks]

    def text_search(
        self,
        query_text: str,
        k: Optional[int] = 5,
        query_maxlen: Optional[int] = None,
        include_embedding: Optional[bool] = False,
        **kwargs: Any,
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieves a list of text chunks relevant to a given query from the vector store, ranked by
        relevance or other metrics.

        Parameters:
            query_text (str): The query text to search for relevant text chunks.
            k (Optional[int]): The number of top results to retrieve. Default 5.
            query_maxlen (Optional[int]): The maximum length of the query to consider. If None, the
                                          maxlen will be dynamically generated.
            include_embedding (Optional[bool]): Optional (default False) flag to include the
                                                embedding vectors in the returned chunks
            **kwargs (Any): Additional parameters that implementations might require for customized
                            retrieval operations.

        Returns:
            List[Tuple[Chunk, float]]: A list of retrieved Chunk, float Tuples, each representing a text chunk that is relevant
                                  to the query, along with its similarity score.
        """

        return asyncio.run(
            self.atext_search(
                query_text=query_text,
                k=k,
                query_maxlen=query_maxlen,
                include_embedding=include_embedding,
            )
        )

    def embedding_search(
        self,
        query_embedding: Embedding,
        k: Optional[int] = 5,
        include_embedding: Optional[bool] = False,
        **kwargs: Any,
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieves a list of text chunks relevant to a given query from the vector store, ranked by
        relevance or other metrics.

        Parameters:
            query_embedding (Embedding): The query embedding to search for relevant text chunks.
            k (Optional[int]): The number of top results to retrieve. Default 5.
            include_embedding (Optional[bool]): Optional (default False) flag to include the
                                                embedding vectors in the returned chunks
            **kwargs (Any): Additional parameters that implementations might require for customized
                            retrieval operations.

        Returns:
            List[Tuple[Chunk, float]]: A list of retrieved Chunk, float Tuples, each representing a text chunk that is relevant
                                  to the query, along with its similarity score.
        """

        return asyncio.run(
            self.aembedding_search(
                query_embedding=query_embedding,
                k=k,
                include_embedding=include_embedding,
            )
        )
