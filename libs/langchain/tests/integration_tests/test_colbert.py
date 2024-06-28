import logging
from typing import List, Tuple

import pytest
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from ragstack_colbert import CassandraDatabase, ColbertEmbeddingModel
from ragstack_langchain.colbert import ColbertVectorStore
from ragstack_tests_utils import TestData
from transformers import BertTokenizer

logging.getLogger("cassandra").setLevel(logging.ERROR)

from tests.integration_tests.conftest import (
    get_astradb_test_store,
    get_local_cassandra_test_store,
)

test_data = {
    "chunks": None,
}


def get_test_chunks() -> List[Document]:
    if test_data["chunks"] is None:
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

        chunk_size = 250
        chunk_overlap = 50

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        def _len_function(text: str) -> int:
            return len(tokenizer.tokenize(text))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=_len_function,
        )

        test_data["chunks"] = text_splitter.split_documents(docs)
        logging.info(
            f"split {len(docs)} documents into {len(test_data['chunks'])} chunks"
        )

    return test_data["chunks"]


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
def test_sync_from_docs(request, vector_store: str):
    vector_store = request.getfixturevalue(vector_store)
    session = vector_store.create_cassandra_session()
    session.default_timeout = 180

    table_name = "LangChain_test_sync_from_docs"

    database = CassandraDatabase.from_session(session=session, table_name=table_name)

    batch_size = 5  # 640 recommended for production use
    chunk_size = 250

    embedding_model = ColbertEmbeddingModel(
        doc_maxlen=chunk_size,
        chunk_batch_size=batch_size,
    )

    logging.debug("Starting to embed ColBERT docs and save them to the database")

    doc_chunks: List[Document] = get_test_chunks()
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

    retriever = vector_store.as_retriever(k=2)
    results: List[Document] = retriever.invoke(
        "What role do coral reefs play in marine ecosystems?"
    )
    assert validate_retrieval(results, key_value="coral reefs")


@pytest.mark.parametrize("vector_store", ["cassandra", "astra_db"])
@pytest.mark.asyncio
async def test_async_from_docs(request, vector_store: str):
    vector_store = request.getfixturevalue(vector_store)
    session = vector_store.create_cassandra_session()
    session.default_timeout = 180

    table_name = "LangChain_test_async_from_docs"

    database = CassandraDatabase.from_session(session=session, table_name=table_name)

    batch_size = 5  # 640 recommended for production use
    chunk_size = 250

    embedding_model = ColbertEmbeddingModel(
        doc_maxlen=chunk_size,
        chunk_batch_size=batch_size,
    )

    logging.debug("Starting to embed ColBERT docs and save them to the database")

    doc_chunks: List[Document] = get_test_chunks()
    vector_store: ColbertVectorStore = await ColbertVectorStore.afrom_documents(
        documents=doc_chunks, database=database, embedding_model=embedding_model
    )

    results: List[Document] = await vector_store.asimilarity_search(
        "What challenges does the Quantum Opacity phenomenon present to the crew of the Nebula Voyager"
    )
    assert validate_retrieval(results, key_value="Quantum Opacity")

    results: List[
        Tuple[Document, float]
    ] = await vector_store.asimilarity_search_with_score(
        "What are Xenospheric Particulates?"
    )

    assert len(results) > 3
    assert results[1][1] > 0  # check score from result 2
    assert results[2][1] > 0  # check score from result 3
    assert results[1][1] > results[2][1]  # check that scores are returned in order

    assert validate_retrieval(
        [r[0] for r in results], key_value="Xenospheric Particulates"
    )

    results: List[Document] = await vector_store.asimilarity_search(
        "What communication methods do dolphins use within their pods?"
    )
    assert validate_retrieval(results, key_value="dolphin")

    retriever = vector_store.as_retriever(k=2)
    results: List[Document] = await retriever.ainvoke(
        "What role do coral reefs play in marine ecosystems?"
    )
    assert validate_retrieval(results, key_value="coral reefs")
