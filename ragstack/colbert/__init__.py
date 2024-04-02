"""
This package provides a suite of tools for encoding and retrieving text using the Colbert model, integrated with a
Cassandra database for scalable storage and retrieval operations. It includes classes for token embeddings,
managing the vector store, and performing efficient similarity searches. Utilities for working with embeddings
and constants related to the Colbert model configuration are also provided.

Exports:
- ColbertTokenEmbeddings: Class for generating and managing token embeddings using the Colbert model.
- CassandraColbertVectorStore: Implementation of a vector store using Cassandra for embedding storage.
- ColbertCassandraRetriever: Retriever class for executing similarity searches within a Cassandra vector store.
- max_similarity_torch: Function for calculating the maximum similarity between embeddings using PyTorch.
- EmbeddedChunk: Data class for representing a chunk of embedded text.
- TokenEmbeddings: Abstract base class for token embedding models.
- ColbertVectorStore: Abstract base class for a vector store.
- DEFAULT_COLBERT_MODEL: The default identifier for the Colbert model.
- DEFAULT_COLBERT_DIM: The default dimensionality for Colbert model embeddings.
"""

from .cassandra_retriever import ColbertCassandraRetriever, max_similarity_torch
from .cassandra_store import CassandraColbertVectorStore
from .colbert_embedding import ColbertTokenEmbeddings
from .constant import DEFAULT_COLBERT_DIM, DEFAULT_COLBERT_MODEL
from .token_embedding import EmbeddedChunk, TokenEmbeddings
from .vector_store import ColbertVectorStore

__all__ = (
    ColbertTokenEmbeddings,
    CassandraColbertVectorStore,
    ColbertCassandraRetriever,
    max_similarity_torch,
    EmbeddedChunk,
    TokenEmbeddings,
    ColbertVectorStore,
    DEFAULT_COLBERT_MODEL,
    DEFAULT_COLBERT_DIM,
)
