from collections.abc import Sequence
from pathlib import Path
from typing import Self

from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.pydantic_v1 import BaseModel

from .traverse import Node, Relation


class NodeSchema(BaseModel):
    """Schema for a node."""

    type: str
    """The name of the node type."""

    description: str
    """Description of the node type."""


class EdgeSchema(BaseModel):
    """Schema for an edge."""

    type: str
    """The name of the edge type."""

    description: str
    """Description of the edge type."""


class RelationshipSchema(BaseModel):
    """Schema for a relationship."""

    edge_type: str
    """The name of the edge type for the relationhsip."""

    source_types: list[str]
    """The node types for the source of the relationship."""

    target_types: list[str]
    """The node types for the target of the relationship."""

    description: str
    """Description of the relationship."""


class Example(BaseModel):
    """An example of a graph."""

    input: str
    """The source input."""

    nodes: Sequence[Node]
    """The extracted example nodes."""

    edges: Sequence[Relation]
    """The extracted example relationhsips."""


class KnowledgeSchema(BaseModel):
    """Schema for a knowledge graph."""

    nodes: list[NodeSchema]
    """Allowed node types for the knowledge schema."""

    relationships: list[RelationshipSchema]
    """Allowed relationships for the knowledge schema."""

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Load a KnowledgeSchema from a JSON or YAML file.

        Args:
            path: The path to the file to load.
        """
        from pydantic_yaml import parse_yaml_file_as

        return parse_yaml_file_as(cls, path)  # type: ignore[type-var]

    def to_yaml_str(self) -> str:
        """Convert the schema to a YAML string."""
        from pydantic_yaml import to_yaml_str

        return to_yaml_str(self)  # type: ignore[arg-type]


class KnowledgeSchemaValidator:
    """Validates graph documents against a knowledge schema."""

    def __init__(self, schema: KnowledgeSchema) -> None:
        self._schema = schema

        self._nodes = {node.type: node for node in schema.nodes}

        self._relationships: dict[str, list[RelationshipSchema]] = {}
        for r in schema.relationships:
            self._relationships.setdefault(r.edge_type, []).append(r)

            # TODO: Validate the relationship.
            # source/target type should exist in nodes, edge_type should exist in edges

    def validate_graph_document(self, document: GraphDocument) -> None:
        """Validate a graph document against the schema."""
        e = ValueError("Invalid graph document for schema")
        for node_type in {node.type for node in document.nodes}:
            if node_type not in self._nodes:
                e.add_note(f"No node type '{node_type}")
        for r in document.relationships:
            relationships = self._relationships.get(r.type, None)
            if relationships is None:
                e.add_note(f"No edge type '{r.type}")
            elif not any(
                candidate
                for candidate in relationships
                if r.source.type in candidate.source_types
                and r.target.type in candidate.target_types
            ):
                e.add_note(
                    "No relationship allows "
                    f"({r.source.id} -> {r.type} -> {r.target.type})"
                )

        if e.__notes__:
            raise e
