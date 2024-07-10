from typing import List

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from ragstack_knowledge_graph.schema_inference import KnowledgeSchemaInferer

MARIE_CURIE_SOURCE = """
Marie Curie, was a Polish and naturalised-French physicist and chemist who
conducted pioneering research on radioactivity field. She was the first woman to win a
Nobel Prize, the first person to win a Nobel Prize twice, and the only person to
win a Nobel Prize in two scientific fields. Her husband, Pierre Curie, was a
won first Nobel Prize with her, making them the first-ever married couple to
win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of
Paris.
"""


@pytest.mark.flaky(reruns=5, reruns_delay=0)
def test_schema_inference(llm: BaseChatModel) -> None:
    schema_inferer = KnowledgeSchemaInferer(llm)

    results = schema_inferer.infer_schemas_from(
        [Document(page_content=MARIE_CURIE_SOURCE)]
    )[0]

    print(results.to_yaml_str())
    nodes = [n.type for n in results.nodes]
    print(nodes)
    any_of_in_list(nodes, "person")
    any_of_in_list(nodes, "institution")
    any_of_in_list(nodes, "award")
    any_of_in_list(nodes, "nationality")
    any_of_in_list(nodes, "field_of_study", "field")

    assert len(results.relationships) > 0
    rels = [r.edge_type for r in results.relationships]
    print(rels)
    any_of_in_list(rels, "won", "won_award")
    any_of_in_list(rels, "is_nationality_of", "has_nationality")
    any_of_in_list(
        rels, "first_professor_at", "professor_at", "works_at", "has_position_at"
    )
    any_of_in_list(rels, "conducted_research_in")
    # We don't do more testing here since this is meant to attempt to infer things.


def any_of_in_list(values: List[str], *expected: str) -> None:
    for value in values:
        if value in expected:
            return
    raise AssertionError(f"Expected one of {expected}, but got {values}")
