from typing import Sequence, cast

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from knowledge_graph.knowledge_schema import KnowledgeSchema
from knowledge_graph.templates import load_template


class KnowledgeSchemaInferer:
    def __init__(self, llm: BaseChatModel) -> None:
        prompt = load_template(
            "schema_inference.md",
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate(prompt=load_template("schema_inference.md")),
                HumanMessagePromptTemplate.from_template("Input: {input}"),
            ]
        )
        # TODO: Use "full" output so we can detect parsing errors?
        structured_llm = llm.with_structured_output(KnowledgeSchema)
        self._chain = prompt | structured_llm

    def infer_schemas_from(self, documents: Sequence[Document]) -> Sequence[KnowledgeSchema]:
        responses = self._chain.batch([{"input": doc.page_content} for doc in documents])
        return cast(Sequence[KnowledgeSchema], responses)
