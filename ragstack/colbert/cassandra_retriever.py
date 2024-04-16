"""
This module integrates text embedding retrieval and similarity computation functionalities with a Cassandra
database backend, optimized for high-performance operations in large-scale text retrieval applications.

Note:
The implementation assumes the availability of a GPU for optimal performance but is designed to fallback
to CPU computation if needed. This flexibility ensures that the retrieval system can be deployed in a
variety of hardware environments.
"""

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from cassandra import ReadTimeout
from torch import Tensor

from .cassandra_store import CassandraColbertVectorStore
from .colbert_embedding import ColbertTokenEmbeddings
from .vector_store import ColbertVectorStoreRetriever, RetrievedChunk


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


class ColbertCassandraRetriever(ColbertVectorStoreRetriever):
    """
    A retriever class that implements the retrieval of text chunks from a Cassandra database, based on
    their semantic similarity to a given query. This implementation leverages the ColBERT model for
    generating embeddings of the query.

    Attributes:
        vector_store (CassandraColbertVectorStore): The ColBERT vector store instance for interacting with the
                                                    Cassandra database.
        colbert_embeddings (ColbertTokenEmbeddings): The ColbertTokenEmbeddings instance for encoding queries.
        is_cuda (bool): A flag indicating whether to use CUDA (GPU) for computation.
        is_fp16 (bool): A flag indicating whether to half-precision floating point operations on CUDA (GPU).
                        Has no effect on CPU computation.
        max_workers: The maximum number of concurrent requests to make to the vector store on a per-retrieval basis.

    Note:
        The class is designed to work with a GPU for optimal performance but will automatically fall back to CPU
        computation if a GPU is not available.
    """

    vector_store: CassandraColbertVectorStore
    colbert_embeddings: ColbertTokenEmbeddings
    max_workers: int
    is_cuda: bool = False
    is_fp16: bool = False

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        vector_store: CassandraColbertVectorStore,
        colbert_embeddings: ColbertTokenEmbeddings,
        max_casandra_workers: Optional[int] = 10,
    ):
        """
        Initializes the retriever with a specific vector store and Colbert embeddings model.

        Parameters:
            vector_store (CassandraColbertVectorStore): The vector store to be used for retrieving embeddings.
            colbert_embeddings (ColbertTokenEmbeddings): The ColBERT embeddings model to be used for encoding
                                                         queries.
            max_casandra_workers: The maximum number of concurrent requests to make to Cassandra on a
                                  per-retrieval basis.
        """

        self.vector_store = vector_store
        self.colbert_embeddings = colbert_embeddings
        self.is_cuda = torch.cuda.is_available()
        self.is_fp16 = all_gpus_support_fp16(self.is_cuda)
        self.executor = ThreadPoolExecutor(max_workers=max_casandra_workers)

    def close(self) -> None:
        """
        Closes any open resources held by the retriever.
        """
        self.executor.shutdown(wait=True)

    def _query_relevant_chunks(
        self, query_encodings: Tensor, top_k: int, query_timeout: int
    ) -> Set[Tuple[str, int]]:
        """
        Retrieves the top_k ANN results for each embedded query token.
        """
        chunks: Set[Tuple[str, int]] = set()

        def query_and_process(query_vector: Tensor) -> Set[Tuple[str, int]]:
            local_chunks = set()
            try:
                async_future = self.vector_store.session.execute_async(
                    self.vector_store.query_colbert_ann_stmt,
                    [list(query_vector), top_k],
                    timeout=query_timeout,
                )
                embeddings = async_future.result()
                for embedding in embeddings:
                    local_chunks.add((embedding.doc_id, embedding.chunk_id))
            except ReadTimeout:
                logging.warn(f"Query timeout with params: {query_vector}, {top_k}")
                # Handle the timeout or other potential exceptions as needed
            except Exception as e:
                logging.error(
                    f"Error during query execution or result fetching: {str(e)}"
                )
            return local_chunks

        executor_futures = {
            self.executor.submit(query_and_process, qv): qv for qv in query_encodings
        }
        for executor_future in as_completed(executor_futures):
            query_vector = executor_futures[executor_future]
            try:
                result_chunks = executor_future.result()
                chunks.update(result_chunks)
            except Exception as exc:
                logging.error(
                    f"query_vector: {query_vector} generated an exception: {exc}"
                )

        return chunks

    def _retrieve_chunks(
        self, chunks: Set[Tuple[str, int]], query_timeout: int
    ) -> Dict[Tuple[str, int], List[Tensor]]:
        """
        Retrieves chunk data for a set of doc_id and chunk_id pairs, returning a dictionary mapping each pair to a list of PyTorch tensors.
        """
        chunk_data: Dict[Tuple[str, int], List[Tensor]] = {}

        def fetch_chunk_data(
            doc_id: str, chunk_id: int
        ) -> Tuple[Tuple[str, int], List[Tensor]]:
            try:
                async_future = self.vector_store.session.execute_async(
                    self.vector_store.query_colbert_chunks_stmt,
                    [doc_id, chunk_id],
                    timeout=query_timeout,
                )
                rows = async_future.result()
                return (doc_id, chunk_id), [
                    torch.tensor(row.bert_embedding) for row in rows
                ]
            except ReadTimeout:
                logging.warn(f"Query timeout for doc_id {doc_id}, chunk_id {chunk_id}")
            except Exception as e:
                logging.error(
                    f"Error fetching chunk data for doc_id {doc_id}, chunk_id {chunk_id}: {e}"
                )

        executor_futures = [
            self.executor.submit(fetch_chunk_data, doc_id, chunk_id)
            for doc_id, chunk_id in chunks
        ]
        for executor_future in as_completed(executor_futures):
            try:
                result = executor_future.result()
                if result:
                    doc_chunk_pair, embeddings = result
                    chunk_data[doc_chunk_pair] = embeddings
            except Exception as exc:
                logging.error(f"executor future generated an exception: {exc}")

        return chunk_data

    def _score_chunks(
        self, query_encodings: Tensor, chunk_data: Dict[Tuple[str, int], List[Tensor]]
    ) -> Dict[Tuple[str, int], Tensor]:
        """
        Process the retrieved chunk data to calculate scores.
        """
        chunk_scores = {}
        for doc_chunk_pair, embeddings in chunk_data.items():
            chunk_scores[doc_chunk_pair] = sum(
                max_similarity_torch(
                    query_vector=qv,
                    embedding_list=embeddings,
                    is_cuda=self.is_cuda,
                    is_fp16=self.is_fp16,
                )
                for qv in query_encodings
            )
        return chunk_scores

    def _fetch_chunk_texts(
        self,
        chunks_by_score: List[Tuple[str, int]],
        chunk_scores: Dict[Tuple[str, int], Tensor],
    ) -> List[RetrievedChunk]:
        """
        Fetches texts for each chunk in parallel and ranks them based on scores.

        Parameters:
            chunks_by_score (List[Tuple[str, int]]): List of tuples containing (doc_id, chunk_id) sorted by score.
            chunk_scores (Dict[Tuple[str, int], Tensor]): Dictionary mapping (doc_id, chunk_id) to their respective scores.

        Returns:
            List[RetrievedChunk]: A list of RetrievedChunk objects with populated fields.
        """

        def fetch_text(doc_id: str, chunk_id: int):
            """
            Fetches the text for a given doc_id and chunk_id.

            Returns:
                Tuple containing the future result, doc_id, and chunk_id.
            """
            async_future = self.vector_store.session.execute_async(
                self.vector_store.query_chunk_stmt, [doc_id, chunk_id]
            )
            rows = async_future.result()
            return rows, doc_id, chunk_id

        answers: List[RetrievedChunk] = []

        rank = 1
        executor_futures = {
            self.executor.submit(fetch_text, doc_id, chunk_id): (doc_id, chunk_id)
            for doc_id, chunk_id in chunks_by_score
        }
        for executor_future in as_completed(executor_futures):
            try:
                rows, doc_id, chunk_id = executor_future.result()
                score = chunk_scores[(doc_id, chunk_id)]
                answers.append(
                    RetrievedChunk(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        score=score.item(),  # Ensure score is a scalar if it's a tensor
                        rank=rank,
                        text=rows.one().body,
                    )
                )
                rank += 1
            except Exception as e:
                print(f"Failed to fetch or process chunk ({doc_id}, {chunk_id}): {e}")

        return answers

    def retrieve(
        self,
        query: str,
        k: int = 10,
        query_maxlen: int = 64,
        query_timeout: int = 180,  # seconds
        **kwargs: Any,
    ) -> List[RetrievedChunk]:
        """
        Retrieves a list of text chunks most relevant to the given query, using semantic similarity as the criteria.

        Parameters:
            query (str): The text query for which relevant chunks are to be retrieved.
            k (int, optional): The number of top relevant chunks to retrieve. Defaults to 10.
            query_maxlen (int, optional): //TODO figure out a better description for this parameter, and/or a better name.
            query_timeout (int, optional): The timeout in seconds for query execution. Defaults to 180.
            **kwargs (Any): Additional keyword arguments that can be used for extending functionality.

        Returns:
            List[RetrievedChunk]: A list of RetrievedChunk objects, each representing a text chunk that is relevant
                                  to the query, along with its similarity score and rank.

        Note:
            The actual retrieval process involves encoding the query, performing an ANN search to find relevant
            embeddings, scoring these embeddings for similarity, and retrieving the corresponding text chunks.
        """

        # if the query has fewer than a predefined number of tokens Nq,
        # colbert_embeddings will pad it with BERT special [mast] token up to length Nq.
        query_encodings = self.colbert_embeddings.encode_query(
            query, query_maxlen=query_maxlen
        )

        # the min of query_maxlen is 32
        top_k = max(math.floor(len(query_encodings) / 2), 16)
        logging.debug(f"query length {len(query)} embeddings top_k: {top_k}")

        chunks = self._query_relevant_chunks(
            query_encodings=query_encodings, top_k=top_k, query_timeout=query_timeout
        )

        # score each chunk
        chunk_data = self._retrieve_chunks(chunks=chunks, query_timeout=query_timeout)
        chunk_scores = self._score_chunks(
            query_encodings=query_encodings, chunk_data=chunk_data
        )

        # load the source chunk for the top k documents
        chunks_by_score = sorted(chunk_scores, key=chunk_scores.get, reverse=True)[:k]

        answers = self._fetch_chunk_texts(
            chunks_by_score=chunks_by_score, chunk_scores=chunk_scores
        )
        return answers
