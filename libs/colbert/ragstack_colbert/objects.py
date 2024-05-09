"""
This module defines a set of data classes for handling chunks of text in various stages of
processing within the ColBERT retrieval system.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# LlamaIndex Node (chunk) has ids, text, embedding, metadata
#            VectorStore.add(nodes: List[Node]) -> List[str](ids): embeds texts OUTside add
#                       .delete(id)
#                       .query(embedding) -> Nodes, Scores, Ids

# LangChain Document (doc or chunk) has page_content, metadata
#           VectorStore.add(texts: List[str], metadatas: Optional[List[dict]]) -> List[str](ids): embeds texts INside add
#                      .delete(ids: List[str]): deletes by id
#                      .search(query: str) -> List[Document]: uses retriever to search in store
#                      .as_retriever() -> Retriever

# Define Vector and Embedding types
Vector = List[float]
Embedding = List[Vector]
Metadata = Dict[str, Any]


class Chunk(BaseModel):
    doc_id: str = Field(..., description="id of the parent document", frozen=True)
    chunk_id: int = Field(..., description="id of the chunk", frozen=True, ge=0)
    text: str = Field(default=None, description="text of the chunk")
    metadata: Metadata = Field(
        default_factory=dict, description="flat metadata of the chunk"
    )
    embedding: Optional[Embedding] = Field(
        default=None, description="embedding of the chunk"
    )

    class Config:
        validate_assignment = True

    # Define equality based on doc_id and chunk_id only
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Chunk):
            return (self.doc_id == other.doc_id) and (self.chunk_id == other.chunk_id)
        return False

    # Define less than for ordering
    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Chunk):
            return NotImplemented
        return (self.doc_id, self.chunk_id) < (other.doc_id, other.chunk_id)

    # Allow objects to be hashable - only necessary if you need to use them in sets or as dict keys
    def __hash__(self):
        return hash((self.doc_id, self.chunk_id))
