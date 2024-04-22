"""
This package provides a suite of tools for encoding and retrieving text using the Colbert model, integrated with a
Cassandra database for scalable storage and retrieval operations. It includes classes for token embeddings,
managing the vector store, and performing efficient similarity searches. Utilities for working with embeddings
and constants related to the Colbert model configuration are also provided.

Exports:
- ColbertEmbeddings: Class for generating and managing token embeddings using the Colbert model.
- ColbertVectorStore: Implementation of a vector store using Cassandra for embedding storage.
- ColbertCassandraRetriever: Retriever class for executing similarity searches within a Cassandra vector store.
- max_similarity_torch: Function for calculating the maximum similarity between embeddings using PyTorch.
- EmbeddedChunk: Data class for representing a chunk of embedded text.
- TokenEmbeddings: Abstract base class for token embedding models.
- ColbertVectorStore: Abstract base class for a vector store.
- DEFAULT_COLBERT_MODEL: The default identifier for the Colbert model.
- DEFAULT_COLBERT_DIM: The default dimensionality for Colbert model embeddings.
"""

from .cobert_retriever import ColbertRetriever, max_similarity_torch
from .colbert_store import ColbertVectorStore
from .colbert_embeddings import ColbertEmbeddings
from .constant import DEFAULT_COLBERT_DIM, DEFAULT_COLBERT_MODEL
from .chunks import EmbeddedChunk

__all__ = [
    "ColbertEmbeddings",
    "ColbertRetriever",
    "ColbertVectorStore",
    "DEFAULT_COLBERT_MODEL",
    "DEFAULT_COLBERT_DIM",
    "EmbeddedChunk",
    "max_similarity_torch",
]
