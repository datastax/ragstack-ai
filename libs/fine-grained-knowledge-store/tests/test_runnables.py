from precisely import assert_that, contains_exactly

from knowledge_graph.runnables import extract_entities
from knowledge_graph.traverse import Node


def test_extract_entities(llm):
    extractor = extract_entities(llm)
    assert_that(
        extractor.invoke({"question": "Who is Marie Curie?"}),
        contains_exactly(Node("Marie Curie", "Person")),
    )
