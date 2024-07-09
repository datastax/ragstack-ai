from pathlib import Path
from typing import Dict, List, Sequence, Union

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

    source_types: List[str]
    """The node types for the source of the relationship."""

    target_types: List[str]
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

    nodes: List[NodeSchema]
    """Allowed node types for the knowledge schema."""

    relationships: List[RelationshipSchema]
    """Allowed relationships for the knowledge schema."""

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "KnowledgeSchema":
        """Load a KnowledgeSchema from a JSON or YAML file.

        Args:
            path: The path to the file to load.
        """
        from pydantic_yaml import parse_yaml_file_as

        return parse_yaml_file_as(cls, path)

    def to_yaml_str(self) -> str:
        """Convert the schema to a YAML string."""
        from pydantic_yaml import to_yaml_str

        return to_yaml_str(self)


class KnowledgeSchemaValidator:
    """Validates graph documents against a knowledge schema."""

    def __init__(self, schema: KnowledgeSchema) -> None:
        self._schema = schema

        self._nodes = {node.type: node for node in schema.nodes}

        self._relationships: Dict[str, List[RelationshipSchema]] = {}
        for r in schema.relationships:
            self._relationships.setdefault(r.edge_type, []).append(r)

            # TODO: Validate the relationship.
            # source/target type should exist in nodes, edge_type should exist in edges

    def validate_graph_document(self, document: GraphDocument):
        """Validate a graph document against the schema."""
        e = ValueError("Invalid graph document for schema")
        for node_type in {node.type for node in document.nodes}:
            if node_type not in self._nodes:
                e.add_note(f"No node type '{node_type}")
        for r in document.relationships:
            relationships = self._relationships.get(r.edge_type, None)
            if relationships is None:
                e.add_note(f"No edge type '{r.edge_type}")
            else:
                relationship = next(
                    candidate
                    for candidate in relationships
                    if r.source_type in candidate.source_types
                    if r.target_type in candidate.target_types
                )
                if relationship is None:
                    e.add_note(
                        "No relationship allows "
                        f"({r.source_id} -> {r.type} -> {r.target.type})"
                    )

        if e.__notes__:
            raise e
