import logging
from typing import List

import pytest
from cassandra.cluster import Session
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from ragstack_colbert import CassandraDatabase
from ragstack_langchain.colbert import ColbertVectorStore
from ragstack_langchain.colbert.embedding import TokensEmbeddings
from ragstack_tests_utils import TestData
from transformers import BertTokenizer

logging.getLogger("cassandra").setLevel(logging.ERROR)

test_data = {}


def get_test_chunks() -> List[Document]:
    if "chunks" not in test_data:
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
            "split %s documents into %s chunks", len(docs), len(test_data["chunks"])
        )

    return test_data["chunks"]


def validate_retrieval(results: List[Document], key_value: str) -> bool:
    passed = False
    for result in results:
        if key_value in result.page_content:
            passed = True
    return passed


@pytest.mark.parametrize("session", ["cassandra", "astra_db"], indirect=["session"])
def test_sync_from_docs(session: Session) -> None:
    table_name = "LangChain_test_sync_from_docs"

    database = CassandraDatabase.from_session(session=session, table_name=table_name)

    batch_size = 5  # 640 recommended for production use
    chunk_size = 250

    embedding = TokensEmbeddings.colbert(
        doc_maxlen=chunk_size,
        chunk_batch_size=batch_size,
    )

    logging.debug("Starting to embed ColBERT docs and save them to the database")

    doc_chunks: List[Document] = get_test_chunks()
    vector_store: ColbertVectorStore = ColbertVectorStore.from_documents(
        documents=doc_chunks, database=database, embedding=embedding
    )

    results = vector_store.similarity_search(
        "What challenges does the Quantum Opacity phenomenon present to the crew of "
        "the Nebula Voyager"
    )
    assert validate_retrieval(results, key_value="Quantum Opacity")

    results2 = vector_store.similarity_search_with_score(
        "What are Xenospheric Particulates?"
    )

    assert len(results2) > 3  # noqa: PLR2004
    assert results2[1][1] > 0  # check score from result 2
    assert results2[2][1] > 0  # check score from result 3
    assert results2[1][1] > results2[2][1]  # check that scores are returned in order

    assert validate_retrieval(
        [r[0] for r in results2], key_value="Xenospheric Particulates"
    )

    results3 = vector_store.similarity_search(
        "What communication methods do dolphins use within their pods?"
    )
    assert validate_retrieval(results3, key_value="dolphin")

    retriever = vector_store.as_retriever(k=2)
    results4 = retriever.invoke("What role do coral reefs play in marine ecosystems?")
    assert validate_retrieval(results4, key_value="coral reefs")


@pytest.mark.parametrize("session", ["cassandra", "astra_db"], indirect=["session"])
async def test_async_from_docs(session: Session) -> None:
    table_name = "LangChain_test_async_from_docs"

    database = CassandraDatabase.from_session(session=session, table_name=table_name)

    batch_size = 5  # 640 recommended for production use
    chunk_size = 250

    embedding = TokensEmbeddings.colbert(
        doc_maxlen=chunk_size,
        chunk_batch_size=batch_size,
    )

    logging.debug("Starting to embed ColBERT docs and save them to the database")

    doc_chunks: List[Document] = get_test_chunks()
    vector_store: ColbertVectorStore = await ColbertVectorStore.afrom_documents(
        documents=doc_chunks, database=database, embedding=embedding
    )

    results = await vector_store.asimilarity_search(
        "What challenges does the Quantum Opacity phenomenon present to the crew of "
        "the Nebula Voyager"
    )
    assert validate_retrieval(results, key_value="Quantum Opacity")

    results2 = await vector_store.asimilarity_search_with_score(
        "What are Xenospheric Particulates?"
    )

    assert len(results) > 3  # noqa: PLR2004
    assert results2[1][1] > 0  # check score from result 2
    assert results2[2][1] > 0  # check score from result 3
    assert results2[1][1] > results2[2][1]  # check that scores are returned in order

    assert validate_retrieval(
        [r[0] for r in results2], key_value="Xenospheric Particulates"
    )

    results3 = await vector_store.asimilarity_search(
        "What communication methods do dolphins use within their pods?"
    )
    assert validate_retrieval(results3, key_value="dolphin")

    retriever = vector_store.as_retriever(k=2)
    results4 = await retriever.ainvoke(
        "What role do coral reefs play in marine ecosystems?"
    )
    assert validate_retrieval(results4, key_value="coral reefs")
