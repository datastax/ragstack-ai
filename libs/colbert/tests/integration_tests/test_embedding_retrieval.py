import logging
from typing import List

import pytest
from cassandra.cluster import Session
from ragstack_colbert import (
    CassandraDatabase,
    ColbertEmbeddingModel,
    ColbertVectorStore,
)
from ragstack_tests_utils import TestData


@pytest.mark.parametrize("session", ["cassandra", "astra_db"], indirect=["session"])
def test_embedding_cassandra_retriever(session: Session) -> None:
    narrative = TestData.marine_animals_text()

    # Define the desired chunk size and overlap size
    chunk_size = 450
    overlap_size = 50

    # Function to generate chunks with the specified size and overlap
    def chunk_texts(text: str, chunk_size: int, overlap_size: int) -> List[str]:
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

    k = 5
    chunk_scores = retriever.text_search(
        query_text="what kind fish lives shallow coral reefs", k=k
    )
    assert len(chunk_scores) == k
    for chunk, score in chunk_scores:
        logging.info("got chunk_id %s with score %s", chunk.chunk_id, score)

    best_chunk = chunk_scores[0][0]
    assert len(best_chunk.text) > 0
    logging.info(
        "Highest scoring chunk_id: %s with text: %s",
        best_chunk.chunk_id,
        best_chunk.text,
    )
