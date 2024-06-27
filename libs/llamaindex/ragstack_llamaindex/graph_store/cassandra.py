from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.graph_stores.types import GraphStore, PropertyGraphStore, LabelledNode, Relation, Triplet
from llama_index.core.vector_stores.types import VectorStoreQuery

# class CassandraGraphStore(GraphStore):
#     @property
#     def client(self) -> Any:
#         """Get client."""
#         ...

#     def get(self, subj: str) -> List[List[str]]:
#         """Get triplets."""
#         ...

#     def get_rel_map(
#         self, subjs: Optional[List[str]] = None, depth: int = 2, limit: int = 30
#     ) -> Dict[str, List[List[str]]]:
#         """Get depth-aware rel map."""
#         ...

#     def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
#         """Add triplet."""
#         ...

#     def delete(self, subj: str, rel: str, obj: str) -> None:
#         """Delete triplet."""
#         ...

#     def get_schema(self, refresh: bool = False) -> str:
#         """Get the schema of the graph store."""
#         ...

#     def query(self, query: str, param_map: Optional[Dict[str, Any]] = {}) -> Any:
#         """Query the graph store with statement and parameters."""
#         ...


class CassandraPropertyGraphStore(PropertyGraphStore):
    @property
    def client(self) -> Any:
        """Get client."""


    def get(
        self,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Get nodes with matching values."""

    def get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get triplets with matching values."""


    def get_rel_map(
        self,
        graph_nodes: List[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get depth-aware rel map."""

    def upsert_nodes(self, nodes: List[LabelledNode]) -> None:
        """Upsert nodes."""


    def upsert_relations(self, relations: List[Relation]) -> None:
        """Upsert relations."""

    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Delete matching data."""


    def structured_query(
        self, query: str, param_map: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Query the graph store with statement and parameters."""


    def vector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        """Query the graph store with a vector store query."""

