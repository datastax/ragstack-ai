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
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from cassandra.cluster import ResponseFuture
from torch import Tensor

from .cassandra_store import CassandraColbertVectorStore
from .colbert_embedding import ColbertTokenEmbeddings
from .vector_store import ColbertVectorStoreRetriever, RetrievedChunk


def max_similarity_torch(
    query_vector: Tensor, embedding_list: List[Tensor], is_cuda: Optional[bool] = False
) -> Tensor:
    """
    Calculates the maximum similarity (dot product) between a query vector and a list of embedding vectors,
    leveraging PyTorch for efficient computation.

    Parameters:
        query_vector (Tensor): A 1D tensor representing the query vector.
        embedding_list (List[Tensor]): A list of 1D tensors, each representing an embedding vector.
        is_cuda (Optional[bool]): A flag indicating whether to use CUDA (GPU) for computation. Defaults to False.

    Returns:
        Tensor: A tensor containing the highest similarity score (dot product value) found between the query vector
                and any of the embedding vectors in the list.

    Note:
        This function is designed to run on GPU for enhanced performance but can also execute on CPU.
    """

    # stacks the list of embedding tensors into a single tensor
    if is_cuda:
        query_vector = query_vector.to("cuda")
        _embedding_list = torch.stack(embedding_list).to("cuda")
    else:
        _embedding_list = torch.stack(embedding_list)

    # Calculate the dot products in a vectorized manner on the GPU
    sims = torch.matmul(_embedding_list, query_vector)

    # Find the maximum similarity (dot product) value
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

    Note:
        The class is designed to work with a GPU for optimal performance but will automatically fall back to CPU
        computation if a GPU is not available.
    """

    vector_store: CassandraColbertVectorStore
    colbert_embeddings: ColbertTokenEmbeddings
    is_cuda: bool = False

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        vector_store: CassandraColbertVectorStore,
        colbert_embeddings: ColbertTokenEmbeddings,
    ):
        """
        Initializes the retriever with a specific vector store and Colbert embeddings model.

        Parameters:
            vector_store (CassandraColbertVectorStore): The vector store to be used for retrieving embeddings.
            colbert_embeddings (ColbertTokenEmbeddings): The ColBERT embeddings model to be used for encoding
                                                         queries.
        """

        self.vector_store = vector_store
        self.colbert_embeddings = colbert_embeddings
        self.is_cuda = torch.cuda.is_available()

    def close(self) -> None:
        """
        Closes any open resources held by the retriever.
        """
        pass

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

        top_k = k
        if k < 1:
            top_k = max(math.floor(len(query_encodings) / 2), 16)
        logging.info(f"query length {len(query)} embeddings top_k: {top_k}")

        # find the most relevant chunks
        ann_futures: List[Tuple[Tensor, ResponseFuture]] = []
        for qv in query_encodings:
            # per token based retrieval
            future = self.vector_store.session.execute_async(
                self.vector_store.query_colbert_ann_stmt,
                [list(qv), top_k],
                timeout=query_timeout,
            )
            ann_futures.append((qv, future))

        chunk_candidates: Dict[Tuple[str, int, Tensor]] = {}
        for qv, future in ann_futures:
            rows = future.result()
            for row in rows:
                # pre_score is the dot product between the query vector and the embedding vector
                pre_score = torch.matmul(qv, torch.tensor(row.bert_embedding))
                # should it be previous score + the current score?
                key = (row.doc_id, row.chunk_id, qv)
                chunk_candidates[key] = max(chunk_candidates.get(key, -1), pre_score)

        candidate_scores: Dict[Tuple(str, int)] = {}
        for (doc_id, chunk_id, qv), pre_score in chunk_candidates.items():
            candidate_scores[(doc_id, chunk_id)] = candidate_scores.get((doc_id, chunk_id), 0) + pre_score

        final_candidates = candidate_scores
        if k > 0:
            final_candidates = sorted(candidate_scores, key=candidate_scores.get, reverse=True)[:2*k]

        # score each document
        chunk_scores: Dict[Tuple[str, int], Tensor] = {}
        score_futures: List[Tuple[ResponseFuture, str, int]] = []
        for doc_id, chunk_id in final_candidates:
            future = self.vector_store.session.execute_async(
                self.vector_store.query_colbert_chunks_stmt,
                [doc_id, chunk_id],
                timeout=query_timeout,
            )
            score_futures.append((future, doc_id, chunk_id))

        for future, doc_id, chunk_id in score_futures:
            rows = future.result()
            # find all the found parts so that we can do max similarity search
            embeddings_for_chunk = [torch.tensor(row.bert_embedding) for row in rows]
            # score based on The function returns the highest similarity score
            # (i.e., the maximum dot product value) between the query vector and any of the embedding vectors in the list.
            chunk_scores[(doc_id, chunk_id)] = sum(
                max_similarity_torch(qv, embeddings_for_chunk, self.is_cuda)
                for qv in query_encodings
            )

        # load the source chunk for the top k documents
        rank_k = 10
        if k >= 0:
            rank_k = min(k, 10)
        chunks_by_score = sorted(chunk_scores, key=chunk_scores.get, reverse=True)[:rank_k]

        # grab the chunk texts
        text_futures: List[Tuple[ResponseFuture, str, int]] = []
        for doc_id, chunk_id in chunks_by_score:
            future = self.vector_store.session.execute_async(
                self.vector_store.query_chunk_stmt, [doc_id, chunk_id]
            )
            text_futures.append((future, doc_id, chunk_id))

        answers: List[RetrievedChunk] = []
        rank = 1
        for future, doc_id, chunk_id in text_futures:
            rows = future.result()
            score = chunk_scores[(doc_id, chunk_id)]
            answers.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    score=score.item(),
                    rank=rank,
                    text=rows.one().body,
                )
            )
            rank = rank + 1
        # clean up on tensor memory on GPU
        del chunk_scores
        return answers
