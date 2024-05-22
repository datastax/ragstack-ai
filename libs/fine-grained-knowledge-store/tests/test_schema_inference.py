from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from precisely import assert_that, contains_exactly

from knowledge_graph.schema_inference import KnowledgeSchemaInferer

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


def test_schema_inference(llm: BaseChatModel):
    schema_inferer = KnowledgeSchemaInferer(llm)

    results = schema_inferer.infer_schemas_from([Document(page_content=MARIE_CURIE_SOURCE)])[0]

    print(results.to_yaml_str())
    assert_that(
        [n.type for n in results.nodes],
        contains_exactly("person", "institution", "award", "nationality", "field"),
    )
    assert_that(
        [r.edge_type for r in results.relationships],
        contains_exactly("won", "is_nationality_of", "works_at", "is_field_of"),
    )

    # We don't do more testing here since this is meant to attempt to infer things.
