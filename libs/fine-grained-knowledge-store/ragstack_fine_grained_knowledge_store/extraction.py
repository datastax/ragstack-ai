from typing import Dict, List, Sequence, Union, cast

from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
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

from knowledge_graph.knowledge_schema import (
    Example,
    KnowledgeSchema,
    KnowledgeSchemaValidator,
)
from knowledge_graph.templates import load_template


def _format_example(idx: int, example: Example) -> str:
    from pydantic_yaml import to_yaml_str

    return f"Example {idx}:\n```yaml\n{to_yaml_str(example)}\n```"


class KnowledgeSchemaExtractor:
    def __init__(
        self,
        llm: BaseChatModel,
        schema: KnowledgeSchema,
        examples: Sequence[Example] = [],
        strict: bool = False,
    ) -> None:
        self._validator = KnowledgeSchemaValidator(schema)
        self.strict = strict

        messages = [
            SystemMessagePromptTemplate(
                prompt=load_template("extraction.md", knowledge_schema_yaml=schema.to_yaml_str())
            )
        ]

        if examples:
            formatted = "\n\n".join(map(_format_example, examples))
            messages.append(SystemMessagePromptTemplate(prompt=formatted))

        messages.append(HumanMessagePromptTemplate.from_template("Input: {input}"))

        prompt = ChatPromptTemplate.from_messages(messages)
        schema = create_simple_model(
            node_labels=[node.type for node in schema.nodes],
            rel_types=list({r.edge_type for r in schema.relationships}),
        )
        # TODO: Use "full" output so we can detect parsing errors?
        structured_llm = llm.with_structured_output(schema)
        self._chain = prompt | structured_llm

    def _process_response(
        self, document: Document, response: Union[Dict, BaseModel]
    ) -> GraphDocument:
        raw_graph = cast(_Graph, response)
        nodes = [map_to_base_node(node) for node in raw_graph.nodes] if raw_graph.nodes else []
        relationships = (
            [map_to_base_relationship(rel) for rel in raw_graph.relationships]
            if raw_graph.relationships
            else []
        )

        document = GraphDocument(nodes=nodes, relationships=relationships, source=document)

        if self.strict:
            self._validator.validate_graph_document(document)

        return document

    def extract(self, documents: List[Document]) -> List[GraphDocument]:
        # TODO: Define an async version of extraction?
        responses = self._chain.batch_as_completed(
            [{"input": doc.page_content} for doc in documents]
        )
        return [self._process_response(documents[idx], response) for idx, response in responses]
