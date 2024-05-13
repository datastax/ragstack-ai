import logging
from typing import List, Tuple

import pytest
from langchain_core.documents import Document
from ragstack_colbert import CassandraDatabase, ColbertEmbeddingModel
from ragstack_tests_utils import TestData
from transformers import BertTokenizer

from langchain.text_splitter import RecursiveCharacterTextSplitter
from ragstack_langchain.colbert import ColbertVectorStore

logging.getLogger("cassandra").setLevel(logging.ERROR)

from tests.integration_tests.conftest import (
    get_astradb_test_store,
    get_local_cassandra_test_store,
)


def validate_retrieval(results: List[Document], key_value: str):
    passed = False
    for result in results:
        if key_value in result.page_content:
            passed = True
    return passed


@pytest.fixture
def cassandra():
    return get_local_cassandra_test_store()


@pytest.fixture
def astra_db():
    return get_astradb_test_store()


@pytest.mark.parametrize("vector_store", ["cassandra", "astra_db"])
def test_sync(request, vector_store: str):
    vector_store = request.getfixturevalue(vector_store)
    session = vector_store.create_cassandra_session()
    session.default_timeout = 180

    table_name = f"LangChain_colbert_sync"

    database = CassandraDatabase.from_session(session=session, table_name=table_name)

    docs: List[Document] = []
    docs.append(
        Document(
            page_content=TestData.marine_animals_text(),
            metadata={"name": "marine_animals"},
        )
    )
    docs.append(
        Document(
            page_content=TestData.nebula_voyager_text(),
            metadata={"name": "nebula_voyager"},
        )
    )

    batch_size = 5  # 640 recommended for production use
    chunk_size = 250
    chunk_overlap = 50

    embedding_model = ColbertEmbeddingModel(doc_maxlen=chunk_size)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def _len_function(text: str) -> int:
        return len(tokenizer.tokenize(text))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_len_function,
    )

    doc_chunks: List[Document] = text_splitter.split_documents(docs)

    logging.info(f"split {len(docs)} documents into {len(doc_chunks)} chunks")

    embedding_model = ColbertEmbeddingModel(
        doc_maxlen=chunk_size,
        chunk_batch_size=batch_size,
    )

    logging.debug("Starting to embed ColBERT docs and save them to the database")

    vector_store: ColbertVectorStore = ColbertVectorStore.from_documents(
        documents=doc_chunks, database=database, embedding_model=embedding_model
    )

    results: List[Document] = vector_store.similarity_search(
        "What challenges does the Quantum Opacity phenomenon present to the crew of the Nebula Voyager"
    )
    assert validate_retrieval(results, key_value="Quantum Opacity")

    results: List[Tuple[Document, float]] = vector_store.similarity_search_with_score(
        "What are Xenospheric Particulates?"
    )

    assert len(results) > 3
    assert results[1][1] > 0  # check score from result 2
    assert results[2][1] > 0  # check score from result 3
    assert results[1][1] > results[2][1]  # check that scores are returned in order

    assert validate_retrieval(
        [r[0] for r in results], key_value="Xenospheric Particulates"
    )

    results: List[Document] = vector_store.similarity_search(
        "What communication methods do dolphins use within their pods?"
    )
    assert validate_retrieval(results, key_value="dolphin")
