from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_experimental.graph_transformers.llm import (
    _Graph,
    create_simple_model,
    map_to_base_node,
    map_to_base_relationship,
)

from .knowledge_schema import (
    Example,
    KnowledgeSchema,
    KnowledgeSchemaValidator,
)
from .templates import load_template

if TYPE_CHECKING:
    from langchain_core.prompts.chat import MessageLikeRepresentation


def _format_example(idx: int, example: Example) -> str:
    from pydantic_yaml import to_yaml_str

    yaml_example = to_yaml_str(example)  # type: ignore[arg-type]
    return f"Example {idx}:\n```yaml\n{yaml_example}\n```"


class KnowledgeSchemaExtractor:
    """Extracts knowledge graphs from documents."""

    def __init__(
        self,
        llm: BaseChatModel,
        schema: KnowledgeSchema,
        examples: Sequence[Example] = (),
        strict: bool = False,
    ) -> None:
        self._validator = KnowledgeSchemaValidator(schema)
        self.strict = strict

        messages: list[MessageLikeRepresentation] = [
            SystemMessagePromptTemplate(
                prompt=load_template(
                    "extraction.md", knowledge_schema_yaml=schema.to_yaml_str()
                )
            )
        ]

        if examples:
            formatted = "\n\n".join(
                [_format_example(i, example) for i, example in enumerate(examples)]
            )
            messages.append(SystemMessage(content=formatted))

        messages.append(HumanMessagePromptTemplate.from_template("Input: {input}"))

        prompt = ChatPromptTemplate.from_messages(messages)
        model_schema = create_simple_model(
            node_labels=[node.type for node in schema.nodes],
            rel_types=list({r.edge_type for r in schema.relationships}),
        )
        # TODO: Use "full" output so we can detect parsing errors?
        structured_llm = llm.with_structured_output(model_schema)
        self._chain = prompt | structured_llm

    def _process_response(
        self, document: Document, response: dict[str, Any] | BaseModel
    ) -> GraphDocument:
        raw_graph = cast(_Graph, response)
        nodes = (
            [map_to_base_node(node) for node in raw_graph.nodes]
            if raw_graph.nodes
            else []
        )
        relationships = (
            [map_to_base_relationship(rel) for rel in raw_graph.relationships]
            if raw_graph.relationships
            else []
        )

        graph_document = GraphDocument(
            nodes=nodes, relationships=relationships, source=document
        )

        if self.strict:
            self._validator.validate_graph_document(graph_document)

        return graph_document

    def extract(self, documents: list[Document]) -> list[GraphDocument]:
        """Extract knowledge graphs from a list of documents."""
        # TODO: Define an async version of extraction?
        responses = self._chain.batch_as_completed(
            [{"input": doc.page_content} for doc in documents]
        )
        return [
            self._process_response(documents[idx], response)
            for idx, response in responses
        ]
