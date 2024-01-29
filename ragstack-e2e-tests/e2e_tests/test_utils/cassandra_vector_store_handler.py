import logging
import os
from typing import List

import cassio
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from cassio.table import MetadataVectorCassandraTable
from langchain_community.chat_message_histories import (
    CassandraChatMessageHistory,
)
from langchain_community.vectorstores.cassandra import Cassandra
from langchain_core.chat_history import BaseChatMessageHistory
from llama_index.schema import TextNode
from llama_index.vector_stores import CassandraVectorStore
from llama_index.vector_stores.types import (
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
    cassandra_container = None

    def __init__(self, implementation: VectorStoreImplementation) -> None:
        super().__init__(implementation, [VectorStoreImplementation.CASSANDRA])
        self.cassandra_session = None
        self.test_table_name = None

    def before_test(self) -> VectorStoreTestContext:
        super().check_implementation()

        self.test_table_name = "table_" + random_string()

        start_container = os.environ.get("CASSANDRA_START_CONTAINER", "true")
        if start_container == "true":
            if CassandraVectorStoreHandler.cassandra_container is None:
                CassandraVectorStoreHandler.cassandra_container = CassandraContainer()
                CassandraVectorStoreHandler.cassandra_container.start()
                logging.info("Cassandra container started")
            else:
                logging.info("Cassandra container already started")
            cassandra_port = (
                CassandraVectorStoreHandler.cassandra_container.get_mapped_port()
            )
        else:
            logging.info("Connecting to local Cassandra instance")
            cassandra_port = 9042

        cluster = Cluster(
            [("127.0.0.1", cassandra_port)],
            auth_provider=PlainTextAuthProvider("cassandra", "cassandra"),
        )
        self.cassandra_session = cluster.connect()
        keyspace = "default_keyspace"
        self.cassandra_session.execute(f"DROP KEYSPACE IF EXISTS {keyspace}")
        self.cassandra_session.execute(
            f"CREATE KEYSPACE IF NOT EXISTS {keyspace} WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': '1'}}"
        )
        cassio.init(session=self.cassandra_session)
        return CassandraVectorStoreTestContext(self)


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
        if isinstance(self.table, MetadataVectorCassandraTable):
            results = self.table.ann_search(vector=vector, n=limit)
            docs = []
            for result in results:
                docs.append(result["body_blob"])
            return docs
        else:
            results = self.table.search(embedding_vector=vector, top_k=limit)
            docs = []
            for result in results:
                docs.append(result["document"])
            return docs


class EnhancedCassandraLlamaIndexVectorStore(
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

    def new_langchain_vector_store(
        self, **kwargs
    ) -> EnhancedCassandraLangChainVectorStore:
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

    def new_llamaindex_vector_store(
        self, **kwargs
    ) -> EnhancedCassandraLlamaIndexVectorStore:
        return EnhancedCassandraLlamaIndexVectorStore(
            session=self.handler.cassandra_session,
            keyspace="default_keyspace",
            table=self.handler.test_table_name,
            **kwargs,
        )
