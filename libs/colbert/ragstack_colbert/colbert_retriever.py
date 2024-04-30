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
import nest_asyncio
from typing import Any, Dict, List, Optional, Set

import torch
from torch import Tensor

from .base_embedding_model import BaseEmbeddingModel
from .base_retriever import BaseRetriever
from .base_vector_store import BaseVectorStore
from .objects import BaseChunk, ChunkData, RetrievedChunk


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
    query_vector: Tensor,
    embedding_list: List[Tensor],
    is_cuda: Optional[bool] = False,
    is_fp16: Optional[bool] = False,
) -> Tensor:
    """
    Calculates the maximum similarity (dot product) between a query vector and a list of embedding vectors,
    leveraging PyTorch for efficient computation.

    Parameters:
        query_vector (Tensor): A 1D tensor representing the query vector.
        embedding_list (List[Tensor]): A list of 1D tensors, each representing an embedding vector.
        is_cuda (Optional[bool]): A flag indicating whether to use CUDA (GPU) for computation. Defaults to False.
        is_fp16 (bool): A flag indicating whether to half-precision floating point operations on CUDA (GPU).
                        Has no effect on CPU computation. Defaults to False.

    Returns:
        Tensor: A tensor containing the highest similarity score (dot product value) found between the query vector
                and any of the embedding vectors in the list.

    Note:
        This function is designed to run on GPU for enhanced performance but can also execute on CPU.
    """

    # Convert embedding list to a tensor
    embedding_tensor = torch.stack(embedding_list)

    if is_cuda:
        device = torch.device("cuda")
        query_vector = query_vector.to(device)
        embedding_tensor = embedding_tensor.to(device)

        # Use half-precision operations if supported
        if is_fp16:
            query_vector = query_vector.half()
            embedding_tensor = embedding_tensor.half()

    # Perform the dot product operation
    sims = torch.matmul(embedding_tensor, query_vector)

    # Find the maximum similarity
    max_sim = torch.max(sims)

    # returns a tensor; the item() is the score
    return max_sim

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

    vector_store: BaseVectorStore
    embedding_model: BaseEmbeddingModel
    is_cuda: bool = False
    is_fp16: bool = False

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_model: BaseEmbeddingModel,
    ):
        """
        Initializes the retriever with a specific vector store and Colbert embeddings model.

        Parameters:
            vector_store (BaseVectorStore): The vector store to be used for retrieving embeddings.
            embedding_model (BaseEmbeddingModel): The ColBERT embeddings model to be used for encoding
                                                         queries.
        """

        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.is_cuda = torch.cuda.is_available()
        self.is_fp16 = all_gpus_support_fp16(self.is_cuda)

    def close(self) -> None:
        """
        Closes any open resources held by the retriever.
        """
        pass

    async def _query_relevant_chunks(
        self, query_embeddings: List[Tensor], top_k: int
    ) -> Set[BaseChunk]:
        """
        Retrieves the top_k ANN results for each embedded query token.
        """
        chunks: Set[BaseChunk] = set()
        # Collect all tasks
        tasks = [self.vector_store.search_relevant_chunks(vector=v, n=top_k) for v in query_embeddings]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle potential exceptions
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Issue on vector_store.get_relevant_chunks(): {result} at {get_trace(result)}")
            else:
                chunks.update(result)

        return chunks

    async def _retrieve_chunks(
        self, chunks: Set[BaseChunk]
    ) -> Dict[BaseChunk, List[Tensor]]:
        """
        Retrieves embeddings for a list of chunks, returning a dictionary mapping chunk to a list of PyTorch tensors.
        """
        chunk_embeddings: Dict[BaseChunk, List[Tensor]] = {}

        # Collect all tasks
        tasks = [self.vector_store.get_chunk_embeddings(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle potential exceptions
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Issue on vector_store.get_chunk_embeddings(): {result} at {get_trace(result)}")
            else:
                chunk, embeddings = result
                chunk_embeddings[chunk] = embeddings

        return chunk_embeddings

    def _score_chunks(
        self, query_embeddings: Tensor, chunk_data: Dict[BaseChunk, List[Tensor]]
    ) -> Dict[BaseChunk, Tensor]:
        """
        Process the retrieved chunk data to calculate scores.
        """
        chunk_scores = {}
        for chunk, embeddings in chunk_data.items():
            chunk_scores[chunk] = sum(
                max_similarity_torch(
                    query_vector=qv,
                    embedding_list=embeddings,
                    is_cuda=self.is_cuda,
                    is_fp16=self.is_fp16,
                )
                for qv in query_embeddings
            )
        return chunk_scores

    async def _fetch_chunk_data(
        self,
        chunks_by_score: List[BaseChunk],
        chunk_scores:  Dict[BaseChunk, Tensor],
    ) -> List[RetrievedChunk]:
        """
        Fetches text and metadata for each chunk and ranks them based on scores.

        Parameters:
            chunks_by_score (List[Tuple[str, int]]): List of tuples containing (doc_id, chunk_id) sorted by score.
            chunk_scores (Dict[Tuple[str, int], Tensor]): Dictionary mapping (doc_id, chunk_id) to their respective scores.

        Returns:
            List[RetrievedChunk]: A list of RetrievedChunk objects with populated fields.
        """

        # Collect all tasks
        tasks = [self.vector_store.get_chunk_data(chunk=chunk) for chunk in chunks_by_score]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle potential exceptions
        chunk_data_map: Dict[BaseChunk, ChunkData] = {}
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Issue on vector_store.get_chunk_text_and_metadata(): {result} at {get_trace(result)}")
            else:
                chunk, chunk_data = result
                chunk_data_map[chunk] = chunk_data

        answers: List[RetrievedChunk] = []

        for idx, chunk in enumerate(chunks_by_score):
            score = chunk_scores[chunk]
            chunk_data = chunk_data_map[chunk]
            answers.append(
                    RetrievedChunk(
                        doc_id=chunk.doc_id,
                        chunk_id=chunk.chunk_id,
                        score=score.item(),  # Ensure score is a scalar if it's a tensor
                        rank=idx + 1,
                        data=chunk_data,
                    )
                )

        return answers

    async def aretrieve(
        self,
        query: str,
        k: int = 10,
        query_maxlen: int = 64,
        **kwargs: Any,
    ) -> List[RetrievedChunk]:
        """
        Retrieves a list of text chunks most relevant to the given query, using semantic similarity as the criteria.

        Parameters:
            query (str): The text query for which relevant chunks are to be retrieved.
            k (int, optional): The number of top relevant chunks to retrieve. Defaults to 10.
            query_maxlen (int, optional): The maximum number of tokens in the query. If -1, this will be calculated dynamically.
            query_timeout (int, optional): The timeout in seconds for query execution. Defaults to 180.
            **kwargs (Any): Additional keyword arguments that can be used for extending functionality.

        Returns:
            List[RetrievedChunk]: A list of RetrievedChunk objects, each representing a text chunk that is relevant
                                  to the query, along with its similarity score and rank.

        Note:
            The actual retrieval process involves encoding the query, performing an ANN search to find relevant
            embeddings, scoring these embeddings for similarity, and retrieving the corresponding text chunks.
        """

        query_embeddings = self.embedding_model.embed_query(
            query, query_maxlen=query_maxlen
        )

        top_k = max(math.floor(len(query_embeddings) / 2), 16)
        logging.debug(f"based on query length of {len(query_embeddings)} tokens, retrieving {top_k} results per token-embedding")

        chunks = await self._query_relevant_chunks(
            query_embeddings=query_embeddings, top_k=top_k
        )

        # score each chunk
        chunk_data = await self._retrieve_chunks(chunks=chunks)
        chunk_scores = self._score_chunks(
            query_embeddings=query_embeddings, chunk_data=chunk_data
        )

        # load the source chunk for the top k documents
        chunks_by_score = sorted(chunk_scores, key=chunk_scores.get, reverse=True)[:k]

        answers = await self._fetch_chunk_data(
            chunks_by_score=chunks_by_score, chunk_scores=chunk_scores
        )
        return answers

    def retrieve(
        self,
        query: str,
        k: int = 10,
        query_maxlen: int = 64,
        **kwargs: Any,
    ) -> List[RetrievedChunk]:
        """
        Retrieves a list of text chunks most relevant to the given query, using semantic similarity as the criteria.

        Parameters:
            query (str): The text query for which relevant chunks are to be retrieved.
            k (int, optional): The number of top relevant chunks to retrieve. Defaults to 10.
            query_maxlen (int, optional): //TODO figure out a better description for this parameter, and/or a better name.
            **kwargs (Any): Additional keyword arguments that can be used for extending functionality.

        Returns:
            List[RetrievedChunk]: A list of RetrievedChunk objects, each representing a text chunk that is relevant
                                  to the query, along with its similarity score and rank.

        Note:
            The actual retrieval process involves encoding the query, performing an ANN search to find relevant
            embeddings, scoring these embeddings for similarity, and retrieving the corresponding text chunks.
        """
        # nest_asyncio does not a new event loop to be created
        # in the case there is already an event loop such as colab, it's required
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aretrieve(query=query, k=k, query_maxlen=query_maxlen))
