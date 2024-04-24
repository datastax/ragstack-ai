"""
This package provides a suite of tools for encoding and retrieving text using the ColBERT model, integrated with a
Cassandra database for scalable storage and retrieval operations. It includes classes for token embeddings,
managing the vector store, and performing efficient similarity searches. Utilities for working with embeddings
and constants related to the ColBERT model configuration are also provided.

Exports:
- CassandraVectorStore: Implementation of a ColBERT vector store using Cassandra for storage.
- ColbertEmbeddingModel: Class for generating and managing token embeddings using the ColBERT model.
- ColbertRetriever: Retriever class for executing ColBERT searches within a vector store.
- DEFAULT_COLBERT_MODEL: The default identifier for the ColBERT model.
- DEFAULT_COLBERT_DIM: The default dimensionality for ColBERT model embeddings.
- EmbeddedChunk: Data class for representing a chunk of embedded text.
- RetrievedChunk: Data class for representing a chunk of retrieved text.
"""

from .cassandra_vector_store import CassandraVectorStore
from .colbert_retriever import ColbertRetriever
from .colbert_embedding_model import ColbertEmbeddingModel
from .constant import DEFAULT_COLBERT_DIM, DEFAULT_COLBERT_MODEL
from .objects import ChunkData, EmbeddedChunk, RetrievedChunk

__all__ = [
    "CassandraVectorStore",
    "ChunkData",
    "ColbertEmbeddingModel",
    "ColbertRetriever",
    "DEFAULT_COLBERT_DIM",
    "DEFAULT_COLBERT_MODEL",
    "EmbeddedChunk",
    "RetrievedChunk",
]
