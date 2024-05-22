from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_experimental.graph_transformers.llm import optional_enum_field

from .traverse import Node

QUERY_ENTITY_EXTRACT_PROMPT = (
    "A question is provided below. Given the question, extract up to 5 "
    "entities (name and type) from the text. Focus on extracting the entities "
    " that we can use to best lookup answers to the question. Avoid stopwords.\n"
    "---------------------\n"
    "{question}\n"
    "---------------------\n"
    "{format_instructions}\n"
)


# TODO: Use a knowledge schema when extracting entities, to get the right kinds of nodes.
def extract_entities(
    llm: BaseChatModel,
    keyword_extraction_prompt: str = QUERY_ENTITY_EXTRACT_PROMPT,
    node_types: Optional[List[str]] = None,
) -> Runnable:
    """
    Return a keyword-extraction runnable.

    This will expect a dictionary containing the `"question"` to extract keywords from.

    Parameters:
    - llm: The LLM to use for extracting entities.
    - node_types: List of node types to extract.
    - keyword_extraction_prompt: The prompt to use for requesting entities.
      This should include the `{question}` being asked as well as the `{format_instructions}`
      which describe how to produce the output.
    """
    prompt = ChatPromptTemplate.from_messages([keyword_extraction_prompt])
    assert "question" in prompt.input_variables
    assert "format_instructions" in prompt.input_variables

    class SimpleNode(BaseModel):
        """Represents a node in a graph with associated properties."""

        id: str = Field(description="Name or human-readable unique identifier.")
        type: str = optional_enum_field(node_types, description="The type or label of the node.")

    class SimpleNodeList(BaseModel):
        """Represents a list of simple nodes."""

        nodes: List[SimpleNode]

    output_parser = JsonOutputParser(pydantic_object=SimpleNodeList)
    return (
        RunnablePassthrough.assign(
            format_instructions=lambda _: output_parser.get_format_instructions(),
        )
        | ChatPromptTemplate.from_messages([keyword_extraction_prompt])
        | llm
        | output_parser
        | RunnableLambda(lambda node_list: [Node(n["id"], n["type"]) for n in node_list["nodes"]])
    )
