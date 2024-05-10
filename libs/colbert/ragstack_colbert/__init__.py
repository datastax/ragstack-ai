"""
This package provides a suite of tools for encoding and retrieving text using the ColBERT model, integrated with a
Cassandra database for scalable storage and retrieval operations. It includes classes for token embeddings,
managing the vector store, and performing efficient similarity searches. Utilities for working with embeddings
and constants related to the ColBERT model configuration are also provided.

Exports:
- CassandraDatabase: Implementation of a BaseDatabase using Cassandra for storage.
- ColbertEmbeddingModel: Class for generating and managing token embeddings using the ColBERT model.
- ColbertVectorStore: Implementation of a BaseVectorStore.
- ColbertRetriever: Retriever class for executing ColBERT searches within a vector store.
- DEFAULT_COLBERT_MODEL: The default identifier for the ColBERT model.
- DEFAULT_COLBERT_DIM: The default dimensionality for ColBERT model embeddings.
- Chunk: Data class for representing a chunk of embedded text.
"""

from .cassandra_database import CassandraDatabase
from .colbert_embedding_model import ColbertEmbeddingModel
from .colbert_retriever import ColbertRetriever
from .colbert_vector_store import ColbertVectorStore
from .constant import DEFAULT_COLBERT_DIM, DEFAULT_COLBERT_MODEL
from .objects import Chunk, Embedding, Metadata, Vector

__all__ = [
    "CassandraDatabase",
    "ColbertEmbeddingModel",
    "ColbertRetriever",
    "ColbertVectorStore",
    "DEFAULT_COLBERT_DIM",
    "DEFAULT_COLBERT_MODEL",
    "Chunk",
    "Embedding",
    "Metadata",
    "Vector",
]
