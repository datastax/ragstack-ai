import logging
from typing import Dict, List, Tuple

import pytest
from cassandra.cluster import Session
from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import Document, NodeWithScore, QueryBundle
from llama_index.core.text_splitter import SentenceSplitter
from ragstack_colbert import (
    CassandraDatabase,
    ColbertEmbeddingModel,
    ColbertVectorStore,
    Metadata,
)
from ragstack_llamaindex.colbert import ColbertRetriever
from ragstack_tests_utils import TestData

logging.getLogger("cassandra").setLevel(logging.ERROR)


def validate_retrieval(results: List[NodeWithScore], key_value: str) -> bool:
    passed = False
    for result in results:
        if key_value in result.text:
            passed = True
    return passed


@pytest.mark.parametrize("session", ["astra_db"], indirect=["session"])  # "cassandra",
def test_sync(session: Session) -> None:
    table_name = "LlamaIndex_colbert_sync"

    batch_size = 5  # 640 recommended for production use
    chunk_size = 256
    chunk_overlap = 50

    database = CassandraDatabase.from_session(session=session, table_name=table_name)
    embedding_model = ColbertEmbeddingModel(
        doc_maxlen=chunk_size,
        chunk_batch_size=batch_size,
    )

    vector_store = ColbertVectorStore(
        database=database,
        embedding_model=embedding_model,
    )

    docs = []
    docs.append(
        Document(
            text=TestData.marine_animals_text(), extra_info={"name": "marine_animals"}
        )
    )
    docs.append(
        Document(
            text=TestData.nebula_voyager_text(), extra_info={"name": "nebula_voyager"}
        )
    )

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pipeline = IngestionPipeline(transformations=[splitter])

    nodes = pipeline.run(documents=docs)

    docs2: Dict[str, Tuple[List[str], List[Metadata]]] = {}

    for node in nodes:
        doc_id = node.metadata["name"]
        if doc_id not in docs2:
            docs2[doc_id] = ([], [])
        docs2[doc_id][0].append(node.text)
        docs2[doc_id][1].append(node.metadata)

    logging.debug("Starting to embed ColBERT docs and save them to the database")

    for doc_id in docs2:
        texts = docs2[doc_id][0]
        metadatas = docs2[doc_id][1]

        logging.debug("processing %s that has %s chunks", doc_id, len(texts))

        vector_store.add_texts(texts=texts, metadatas=metadatas, doc_id=doc_id)

    retriever = ColbertRetriever(
        retriever=vector_store.as_retriever(), similarity_top_k=5
    )

    Settings.llm = MockLLM()

    response_synthesizer = get_response_synthesizer()

    pipeline2 = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    results = pipeline2.retrieve(QueryBundle("Who developed the Astroflux Navigator?"))
    assert validate_retrieval(results, key_value="Astroflux Navigator")

    results = pipeline2.retrieve(
        QueryBundle("Describe the phenomena known as 'Chrono-spatial Echoes'")
    )
    assert validate_retrieval(results, key_value="Chrono-spatial Echoes")

    results = pipeline2.retrieve(
        QueryBundle("How do anglerfish adapt to the deep ocean's darkness?")
    )
    assert validate_retrieval(results, key_value="anglerfish")
