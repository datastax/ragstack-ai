from precisely import assert_that, contains_exactly

from ragstack_fine_grained_knowledge_store.runnables import extract_entities
from ragstack_fine_grained_knowledge_store.traverse import Node


def test_extract_entities(llm):
    extractor = extract_entities(llm)
    assert_that(
        extractor.invoke({"question": "Who is Marie Curie?"}),
        contains_exactly(Node("Marie Curie", "Person")),
    )
