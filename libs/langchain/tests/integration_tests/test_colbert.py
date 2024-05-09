import logging
import datetime
import pytest

from typing import List

from ragstack_colbert import CassandraDatabase, ColbertEmbeddingModel
from ragstack_langchain.colbert import ColbertVectorStore

from ragstack_tests_utils import TestData

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import BertTokenizer


from tests.integration_tests.conftest import (
    get_astradb_test_store,
    get_local_cassandra_test_store,
)

@pytest.fixture
def cassandra():
    return get_local_cassandra_test_store()


@pytest.fixture
def astra_db():
    return get_astradb_test_store()

#@pytest.mark.parametrize("vector_store", ["cassandra", "astra_db"])
@pytest.mark.parametrize("vector_store", ["astra_db"])
def test_sync(request, vector_store: str):
    vector_store = request.getfixturevalue(vector_store)
    session=vector_store.create_cassandra_session()
    session.default_timeout = 180

    now = datetime.datetime.now()
    table_name = f"LangChain_colbert_sync_{now.strftime('%Y%m%d_%H%M%S')}"

    database = CassandraDatabase.from_session(session=session, table_name=table_name)

    docs: List[Document] = []
    docs.append(Document(page_content=TestData.marine_animals_text(), metadata={"name": "marine_animals"}))
    docs.append(Document(page_content=TestData.nebula_voyager_text(), metadata={"name": "nebula_voyager"}))

    batch_size = 5 # 640 recommended for production use
    chunk_size = 250
    chunk_overlap = 50

    embedding_model = ColbertEmbeddingModel(doc_maxlen=chunk_size)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    def _len_function(text: str) -> int:
        return len(tokenizer.tokenize(text))


    embedding_model = ColbertEmbeddingModel()



    vector_store: ColbertVectorStore = ColbertVectorStore.from_documents(documents=docs, database=database, embedding_model=embedding_model)

    results = vector_store.search("Who developed the Astroflux Navigator")

    print(results)

