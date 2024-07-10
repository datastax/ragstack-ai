"""Temporary backward-compatibility for KnowledgeStore."""

from .embedding_model import EmbeddingModel
from .graph_store import (
    GraphStore as KnowledgeStore,
)
from .graph_store import (
    Node,
    SetupMode,
)

__all__ = [
    "EmbeddingModel",
    "KnowledgeStore",
    "Node",
    "SetupMode",
]
