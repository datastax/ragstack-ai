import logging

import pytest
from ragstack_colbert import (
    CassandraDatabase,
    ColbertEmbeddingModel,
    ColbertVectorStore,
)
from ragstack_tests_utils import TestData

from tests.integration_tests.conftest import (
    get_astradb_test_store,
    get_local_cassandra_test_store,
)


@pytest.fixture()
def cassandra():
    return get_local_cassandra_test_store()


@pytest.fixture()
def astra_db():
    return get_astradb_test_store()


@pytest.mark.parametrize("vector_store", ["cassandra", "astra_db"])
def test_embedding_cassandra_retriever(request, vector_store: str):
    vector_store = request.getfixturevalue(vector_store)
    narrative = TestData.marine_animals_text()

    # Define the desired chunk size and overlap size
    chunk_size = 450
    overlap_size = 50

    # Function to generate chunks with the specified size and overlap
    def chunk_texts(text, chunk_size, overlap_size):
        texts = []
        start = 0
        end = chunk_size
        while start < len(text):
            # If this is not the first chunk, move back 'overlap_size' characters
            # to create the overlap
            if start != 0:
                start -= overlap_size
            texts.append(text[start:end])
            start = end
            end += chunk_size
        return texts

    # Generate the chunks based on the narrative
    chunked_texts = chunk_texts(narrative, chunk_size, overlap_size)

    doc_id = "marine_animals"

    session = vector_store.create_cassandra_session()
    session.default_timeout = 180

    database = CassandraDatabase.from_session(
        keyspace="default_keyspace",
        table_name="test_embedding_cassandra_retriever",
        session=session,
    )

    embedding_model = ColbertEmbeddingModel()

    store = ColbertVectorStore(
        database=database,
        embedding_model=embedding_model,
    )

    store.add_texts(texts=chunked_texts, doc_id=doc_id)

    retriever = store.as_retriever()

    chunk_scores = retriever.text_search(
        query_text="what kind fish lives shallow coral reefs", k=5
    )
    assert len(chunk_scores) == 5
    for chunk, score in chunk_scores:
        logging.info(f"got chunk_id {chunk.chunk_id} with score {score}")

    best_chunk = chunk_scores[0][0]
    assert len(best_chunk.text) > 0
    logging.info(
        f"Highest scoring chunk_id: {best_chunk.chunk_id} with text: {best_chunk.text}"
    )
