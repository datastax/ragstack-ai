"""Temporary backward-compatibility for KnowledgeStore"""

from .graph_store import (
    EmbeddingModel,
    GraphStore as KnowledgeStore,
    Node,
    SetupMode,
)

__all__ = [
    "EmbeddingModel",
    "KnowledgeStore",
    "Node",
    "SetupMode",
]
