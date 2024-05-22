from os import path

import pytest
from langchain_community.graphs.graph_document import Node, Relationship
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from precisely import assert_that, contains_exactly

from knowledge_graph.extraction import (
    KnowledgeSchema,
    KnowledgeSchemaExtractor,
)


@pytest.fixture(scope="session")
def extractor(llm: BaseChatModel) -> KnowledgeSchemaExtractor:
    schema = KnowledgeSchema.from_file(
        path.join(path.dirname(__file__), "marie_curie_schema.yaml")
    )
    return KnowledgeSchemaExtractor(
        llm=llm,
        schema=schema,
    )


MARIE_CURIE_SOURCE = """
Marie Curie, was a Polish and naturalised-French physicist and chemist who
conducted pioneering research on radioactivity. She was the first woman to win a
Nobel Prize, the first person to win a Nobel Prize twice, and the only person to
win a Nobel Prize in two scientific fields. Her husband, Pierre Curie, was a
co-winner of her first Nobel Prize, making them the first-ever married couple to
win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of
Paris.
"""


def test_extraction(extractor: KnowledgeSchemaExtractor):
    results = extractor.extract([Document(page_content=MARIE_CURIE_SOURCE)])

    marie_curie = Node(id="Marie Curie", type="Person")
    polish = Node(id="Polish", type="Nationality")
    french = Node(id="French", type="Nationality")
    physicist = Node(id="Physicist", type="Occupation")
    chemist = Node(id="Chemist", type="Occupation")
    nobel_prize = Node(id="Nobel Prize", type="Award")
    pierre_curie = Node(id="Pierre Curie", type="Person")

    # Annoyingly, the LLM seems to upper-case `of`. We probably need some instructions around
    # putting things into standard title case, etc.
    university_of_paris = Node(id="University Of Paris", type="Institution")

    assert_that(
        results[0].nodes,
        contains_exactly(
            marie_curie,
            polish,
            french,
            physicist,
            chemist,
            nobel_prize,
            pierre_curie,
            university_of_paris,
        ),
    )
    assert_that(
        results[0].relationships,
        contains_exactly(
            Relationship(source=marie_curie, target=polish, type="HAS_NATIONALITY"),
            Relationship(source=marie_curie, target=french, type="HAS_NATIONALITY"),
            Relationship(source=marie_curie, target=physicist, type="HAS_OCCUPATION"),
            Relationship(source=marie_curie, target=chemist, type="HAS_OCCUPATION"),
            Relationship(source=marie_curie, target=nobel_prize, type="RECEIVED"),
            Relationship(source=pierre_curie, target=nobel_prize, type="RECEIVED"),
            Relationship(source=marie_curie, target=university_of_paris, type="WORKED_AT"),
            Relationship(source=marie_curie, target=pierre_curie, type="MARRIED_TO"),
            Relationship(source=pierre_curie, target=marie_curie, type="MARRIED_TO"),
        ),
    )
