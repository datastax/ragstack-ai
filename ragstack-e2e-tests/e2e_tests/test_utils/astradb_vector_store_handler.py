import logging
import os
from dataclasses import dataclass
from typing import List

import cassio
from langchain_community.chat_message_histories import AstraDBChatMessageHistory
from langchain_community.vectorstores.astradb import AstraDB
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.vectorstores import VectorStore as LangChainVectorStore
from llama_index.vector_stores import AstraDBVectorStore
from llama_index.vector_stores.types import VectorStore as LlamaIndexVectorStore

from e2e_tests.test_utils import get_required_env, random_string
from e2e_tests.test_utils.vector_store_handler import (
    VectorStoreHandler,
    VectorStoreImplementation,
    VectorStoreTestContext,
    EnhancedLangChainVectorStore,
    EnhancedLlamaIndexVectorStore,
)

from astrapy.db import AstraDB as AstraPyClient


@dataclass
class AstraRef:
    token: str
    api_endpoint: str
    collection: str
    id: str
    env: str


def delete_all_astra_collections(astra_ref: AstraRef):
    """
    Deletes all collections.

    Current AstraDB has a limit of 5 collections, meaning orphaned collections
    will cause subsequent tests to fail if the limit is reached.
    """
    raw_client = AstraPyClient(
        api_endpoint=astra_ref.api_endpoint, token=astra_ref.token
    )
    collections = raw_client.get_collections().get("status").get("collections")
    logging.info(f"Existing collections: {collections}")
    for collection_info in collections:
        logging.info(f"Deleting collection: {collection_info}")
        raw_client.delete_collection(collection_info)


def delete_astra_collection(astra_ref: AstraRef) -> None:
    raw_client = AstraPyClient(
        api_endpoint=astra_ref.api_endpoint, token=astra_ref.token
    )
    raw_client.delete_collection(astra_ref.collection)


class AstraDBVectorStoreHandler(VectorStoreHandler):
    def __init__(self):
        super().__init__(
            [VectorStoreImplementation.ASTRADB, VectorStoreImplementation.CASSANDRA]
        )
        env = os.environ.get("ASTRA_DB_ENV", "prod").lower()
        self.astra_ref = AstraRef(
            token=get_required_env("ASTRA_DB_TOKEN"),
            api_endpoint=get_required_env("ASTRA_DB_ENDPOINT"),
            collection="documents" + random_string(),
            id=get_required_env("ASTRA_DB_ID"),
            env=env,
        )

    def before_test(
        self, implementation: VectorStoreImplementation
    ) -> VectorStoreTestContext:
        super().before_test(implementation)
        delete_astra_collection(self.astra_ref)
        delete_all_astra_collections(self.astra_ref)

        if implementation == VectorStoreImplementation.CASSANDRA:
            # to run cassandra implementation over astra
            if self.astra_ref.env == "dev":
                bundle_url_template = "https://api.dev.cloud.datastax.com/v2/databases/{database_id}/secureBundleURL"
                cassio.init(
                    token=self.astra_ref.token,
                    database_id=self.astra_ref.id,
                    bundle_url_template=bundle_url_template,
                )
            else:
                cassio.init(token=self.astra_ref.token, database_id=self.astra_ref.id)
        return AstraDBVectorStoreTestContext(self)

    def after_test(self, implementation: VectorStoreImplementation):
        delete_astra_collection(self.astra_ref)
        delete_all_astra_collections(self.astra_ref)


class EnhancedAstraDBLangChainVectorStore(EnhancedLangChainVectorStore, AstraDB):
    def put_document(
        self, doc_id: str, document: str, metadata: dict, vector: List[float]
    ) -> None:
        self.collection.insert_one(
            {
                "_id": doc_id,
                "document": document,
                "metadata": metadata or {},
                "$vector": vector,
            }
        )

    def search_documents(self, vector: List[float], limit: int) -> List[str]:
        results = self.collection.vector_find(
            vector,
            limit=limit,
        )
        docs = []
        for result in results:
            docs.append(result["document"])
        return docs


class EnhancedAstraDBLlamaIndexVectorStore(
    EnhancedLlamaIndexVectorStore, AstraDBVectorStore
):
    def put_document(
        self, doc_id: str, document: str, metadata: dict, vector: List[float]
    ) -> None:
        self.client.insert_one(
            {
                "_id": doc_id,
                "document": document,
                "metadata": metadata or {},
                "$vector": vector,
            }
        )

    def search(self, vector: List[float], limit: int) -> List[str]:
        results = self.client.vector_find(
            vector,
            limit=limit,
        )
        docs = []
        for result in results:
            docs.append(result["document"])
        return docs


class AstraDBVectorStoreTestContext(VectorStoreTestContext):
    def __init__(self, handler: AstraDBVectorStoreHandler):
        super().__init__()
        self.handler = handler
        self.test_id = "test_id" + random_string()

    def new_langchain_vector_store(self, **kwargs) -> LangChainVectorStore:
        return EnhancedAstraDBLangChainVectorStore(
            collection_name=self.handler.astra_ref.collection,
            token=self.handler.astra_ref.token,
            api_endpoint=self.handler.astra_ref.api_endpoint,
            **kwargs,
        )

    def new_langchain_chat_memory(self, **kwargs) -> BaseChatMessageHistory:
        return AstraDBChatMessageHistory(
            session_id=self.test_id,
            token=self.handler.astra_ref.token,
            api_endpoint=self.handler.astra_ref.api_endpoint,
            collection_name=self.handler.astra_ref.collection + "_chat_memory",
            **kwargs,
        )

    def new_llamaindex_vector_store(self, **kwargs) -> LlamaIndexVectorStore:
        return EnhancedAstraDBLlamaIndexVectorStore(
            token=self.handler.astra_ref.token,
            api_endpoint=self.handler.astra_ref.api_endpoint,
            collection_name=self.handler.astra_ref.collection + "_chat_memory",
            **kwargs,
        )
