"""Temporary backward-compatibility for KnowledgeStore"""

from .graph_store import (
    EmbeddingModel,
    Node,
    SetupMode,
)
from .graph_store import (
    GraphStore as KnowledgeStore,
)

__all__ = [
    "EmbeddingModel",
    "KnowledgeStore",
    "Node",
    "SetupMode",
]
