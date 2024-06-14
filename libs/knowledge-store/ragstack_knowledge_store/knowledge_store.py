"""Temporary backward-compatibility for KnowledgeStore"""
from .graph_store import (
    CONTENT_ID,
    EmbeddingModel,
    GraphStore as KnowledgeStore,
    Node,
    SetupMode,
    TextNode,
)

__all__ = [
    "CONTENT_ID",
    "EmbeddingModel",
    "KnowledgeStore",
    "Node",
    "SetupMode",
    "TextNode",
]
