from typing import List, Set, Tuple, Any, Dict

from cassandra.cluster import ResponseFuture
import logging
import math
from typing import List, Optional

import torch
from torch import tensor

from .cassandra_store import CassandraColbertVectorStore
import logging
from torch import tensor, Tensor
import torch
import math

from .colbert_embedding import ColbertTokenEmbeddings
from .vector_store import ColbertVectorStoreRetriever, RetrievedChunk

# max similarity between a query vector and a list of embeddings
# The function returns the highest similarity score (i.e., the maximum dot product value)
# between the query vector and any of the embedding vectors in the list.

"""
# The function iterates over each embedding vector (e) in the embeddings.
# For each e, it performs a dot product operation (@) with the query vector (qv).
# The dot product of two vectors is a measure of their similarity. In the context of embeddings,
# a higher dot product value usually indicates greater similarity.
# The max function then takes the highest value from these dot product operations.
# Essentially, it's picking the embedding vector that has the highest similarity to the query vector qv.
def max_similarity_operator_based(qv, embeddings, is_cuda: bool=False):
    if is_cuda:
        # Assuming qv and embeddings are PyTorch tensors
        qv = qv.to('cuda')  # Move qv to GPU
        embeddings = [e.to('cuda') for e in embeddings]  # Move all embeddings to GPU
    return max(qv @ e for e in embeddings)
def max_similarity_numpy_based(query_vector, embedding_list):
    # Convert the list of embeddings into a numpy matrix for vectorized operation
    embedding_matrix = np.vstack(embedding_list)

    # Calculate the dot products in a vectorized manner
    sims = np.dot(embedding_matrix, query_vector)

    # Find the maximum similarity (dot product) value
    max_sim = np.max(sims)

    return max_sim
"""


# this torch based max similarity has the best performance.
# it is at least 20 times faster than dot product operator and numpy based implementation CuDA and CPU
def max_similarity_torch(
    query_vector: Tensor, embedding_list: List[Tensor], is_cuda: Optional[bool] = False
) -> Tensor:
    """
    Calculate the maximum similarity (dot product) between a query vector and a list of embedding vectors,
    optimized for performance using PyTorch for GPU acceleration.

    Parameters:
    - query_vector: A PyTorch tensor representing the query vector.
    - embedding_list: A list of PyTorch tensors, each representing an embedding vector.

    Returns:
    - max_sim: A float representing the highest similarity (dot product) score between the query vector
               and the embedding vectors in the list, computed on the GPU.
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
        self.vector_store = vector_store
        self.colbert_embeddings = colbert_embeddings
        self.is_cuda = torch.cuda.is_available()

    def close(self) -> None:
        pass

    def retrieve(
        self, query: str, k: int = 10, query_maxlen: int = 64, **kwargs
    ) -> List[RetrievedChunk]:
        #
        # if the query has fewer than a predefined number of tokens Nq,
        # colbert_embeddings will pad it with BERT special [mast] token up to length Nq.
        #
        query_encodings = self.colbert_embeddings.encode_query(
            query, query_maxlen=query_maxlen
        )

        # the min of query_maxlen is 32
        top_k = max(math.floor(len(query_encodings) / 2), 16)
        logging.debug(f"query length {len(query)} embeddings top_k: {top_k}")

        # find the most relevant documents
        docparts: Set[Tuple[Any, Any]] = set()
        doc_futures = []
        for qv in query_encodings:
            # per token based retrieval
            doc_future = self.vector_store.session.execute_async(
                self.vector_store.query_colbert_ann_stmt, [list(qv), top_k]
            )
            doc_futures.append(doc_future)

        for future in doc_futures:
            embeddings = future.result()
            docparts.update(
                (embedding.doc_id, embedding.chunk_id) for embedding in embeddings
            )

        # score each document
        scores = {}
        futures = []
        for doc_id, chunk_id in docparts:
            future = self.vector_store.session.execute_async(
                self.vector_store.query_colbert_parts_stmt, [doc_id, chunk_id]
            )
            futures.append((future, doc_id, chunk_id))

        for doc_id, chunk_id in docparts:
            # blocking call until the future is done
            rows = future.result()
            # find all the found parts so that we can do max similarity search
            embeddings_for_part = [tensor(row.bert_embedding) for row in rows]
            # score based on The function returns the highest similarity score
            # (i.e., the maximum dot product value) between the query vector and any of the embedding vectors in the list.
            scores[(doc_id, chunk_id)] = sum(
                max_similarity_torch(qv, embeddings_for_part, self.is_cuda)
                for qv in query_encodings
            )
        # load the source chunk for the top k documents
        chunks_by_score = sorted(scores, key=scores.get, reverse=True)[:k]

        # query the doc body
        doc_futures2: Dict[Tuple[Any, Any], ResponseFuture] = {}
        for doc_id, chunk_id, in chunks_by_score:
            future = self.vector_store.session.execute_async(
                self.vector_store.query_chunk_stmt, [doc_id, chunk_id]
            )
            doc_futures2[(doc_id, chunk_id)] = future

        answers: List[RetrievedChunk] = []
        rank = 1
        for doc_id, chunk_id in chunks_by_score:
            rs = doc_futures2[(doc_id, chunk_id)].result()
            score = scores[(doc_id, chunk_id)]
            answers.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    score=score.item(),
                    rank=rank,
                    text=rs.one().body,
                )
            )
            rank = rank + 1
        # clean up on tensor memory on GPU
        del scores
        return answers
