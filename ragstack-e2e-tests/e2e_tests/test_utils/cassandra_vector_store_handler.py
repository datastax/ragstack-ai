import logging
from typing import List

import cassio
from cassio.table import MetadataVectorCassandraTable
from langchain_community.chat_message_histories import (
    CassandraChatMessageHistory,
)
from langchain_community.vectorstores.cassandra import Cassandra
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.vectorstores import VectorStore as LangChainVectorStore
from llama_index.schema import TextNode
from llama_index.vector_stores import CassandraVectorStore
from llama_index.vector_stores.types import (
    VectorStore as LlamaIndexVectorStore,
    VectorStoreQuery,
)

from e2e_tests.test_utils import (
    random_string,
)
from e2e_tests.test_utils.cassandra_container import CassandraContainer
from e2e_tests.test_utils.vector_store_handler import (
    VectorStoreHandler,
    VectorStoreImplementation,
    VectorStoreTestContext,
    EnhancedLangChainVectorStore,
    EnhancedLlamaIndexVectorStore,
)


class CassandraVectorStoreHandler(VectorStoreHandler):
    def __init__(self):
        super().__init__([VectorStoreImplementation.CASSANDRA])
        self.cassandra_container = None
        self.cassandra_session = None
        self.test_table_name = None

    def before_test(
        self, implementation: VectorStoreImplementation
    ) -> VectorStoreTestContext:
        super().before_test(implementation)

        self.test_table_name = "table_" + random_string()
        if self.cassandra_container is None:
            self.cassandra_container = CassandraContainer()
            self.cassandra_container.start()
            logging.info("Cassandra container started")
        else:
            logging.info("Cassandra container already started")

        self.cassandra_session = self.cassandra_container.create_session()
        cassio.init(session=self.cassandra_session)
        return CassandraVectorStoreTestContext(self)

    def after_test(self, implementation: VectorStoreImplementation):
        pass


class EnhancedCassandraLangChainVectorStore(EnhancedLangChainVectorStore, Cassandra):
    def put_document(
        self, doc_id: str, document: str, metadata: dict, vector: List[float]
    ) -> None:
        if isinstance(self.table, MetadataVectorCassandraTable):
            self.table.put(
                row_id=doc_id,
                body_blob=document,
                vector=vector,
                metadata=metadata or {},
            )
        else:
            self.table.table.put(
                row_id=doc_id,
                body_blob=document,
                vector=vector,
                metadata=metadata or {},
            )

    def search_documents(self, vector: List[float], limit: int) -> List[str]:
        results = self.table.search(embedding_vector=vector, top_k=limit)
        if isinstance(self.table, MetadataVectorCassandraTable):
            docs = []
            for result in results:
                docs.append(result["body_blob"])
            return docs
        else:
            docs = []
            for result in results:
                docs.append(result["document"])
            return docs


class EnhancedAstraDBLlamaIndexVectorStore(
    EnhancedLlamaIndexVectorStore, CassandraVectorStore
):
    def put_document(
        self, doc_id: str, document: str, metadata: dict, vector: List[float]
    ) -> None:
        self.add(
            [TextNode(text=document, metadata=metadata, id_=doc_id, embedding=vector)]
        )

    def search_documents(self, vector: List[float], limit: int) -> List[str]:
        return self.query(
            VectorStoreQuery(query_embedding=vector, similarity_top_k=limit)
        ).ids


class CassandraVectorStoreTestContext(VectorStoreTestContext):
    def __init__(self, handler: CassandraVectorStoreHandler):
        super().__init__()
        self.handler = handler
        self.test_id = "test_id" + random_string()

    def new_langchain_vector_store(self, **kwargs) -> LangChainVectorStore:
        return EnhancedCassandraLangChainVectorStore(
            session=self.handler.cassandra_session,
            keyspace="default_keyspace",
            table_name=self.handler.test_table_name,
            **kwargs,
        )

    def new_langchain_chat_memory(self, **kwargs) -> BaseChatMessageHistory:
        return CassandraChatMessageHistory(
            session_id=self.test_id,
            session=self.handler.cassandra_session,
            keyspace="default_keyspace",
            table_name=self.handler.test_table_name + "_chat_memory",
            **kwargs,
        )

    def new_llamaindex_vector_store(self, **kwargs) -> LlamaIndexVectorStore:
        return EnhancedAstraDBLlamaIndexVectorStore(
            session=self.handler.cassandra_session,
            keyspace="default_keyspace",
            table=self.handler.test_table_name,
            **kwargs,
        )
