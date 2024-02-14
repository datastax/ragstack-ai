import concurrent
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import List, Callable

import cassio
from langchain_community.chat_message_histories import AstraDBChatMessageHistory
from langchain_community.vectorstores.astradb import AstraDB
from langchain_core.chat_history import BaseChatMessageHistory
from llama_index.vector_stores import AstraDBVectorStore

from e2e_tests.test_utils import get_required_env, random_string
from e2e_tests.test_utils.vector_store_handler import (
    VectorStoreHandler,
    VectorStoreImplementation,
    VectorStoreTestContext,
    EnhancedLangChainVectorStore,
    EnhancedLlamaIndexVectorStore,
)
from e2e_tests.test_utils.cassandra_vector_store_handler import (
    EnhancedCassandraLlamaIndexVectorStore,
    EnhancedCassandraLangChainVectorStore,
    CassandraChatMessageHistory,
)

from astrapy.db import AstraDB as AstraPyClient


@dataclass
class AstraRef:
    token: str = field(repr=False)
    api_endpoint: str = field(repr=False)
    collection: str
    id: str = field(repr=False)
    env: str


class DeleteCollectionHandler:
    def __init__(self, delete_function: Callable, max_workers=5):
        self.delete_function = delete_function
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers
        self.semaphore = threading.Semaphore(max_workers)

    def get_current_deletions(self):
        """
        Returns the number of ongoing deletions.
        """
        return self.max_workers - self.semaphore._value

    def await_ongoing_deletions_completed(self):
        """
        Blocks until all ongoing deletions are completed.
        """
        while self.semaphore._value != self.max_workers:
            logging.info(
                f"{self.max_workers - self.semaphore._value} deletions still running, waiting to complete"
            )
            time.sleep(1)
        return

    def run_delete(self, collection: str):
        """
        Runs a delete_collection in the background, blocking if max_workers are already running.
        """
        self.semaphore.acquire()  # Wait for a free thread
        future = self.executor.submit(
            lambda: self._run_and_release(collection),
        )
        return future

    def _run_and_release(self, collection: str):
        """
        Internal wrapper to run the delete function and release the semaphore once done.
        """
        try:
            logging.info(f"deleting collection {collection}")
            self.delete_function(collection)
            logging.info(f"deleted collection {collection}")
        finally:
            self.semaphore.release()

    def shutdown(self, wait=True):
        """
        Shuts down the executor, waiting for tasks to complete if specified.
        """
        self.executor.shutdown(wait=wait)


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
    AstraDBVectorStore, EnhancedLlamaIndexVectorStore
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

    def search_documents(self, vector: List[float], limit: int) -> List[str]:
        results = self.client.vector_find(
            vector,
            limit=limit,
        )
        docs = []
        for result in results:
            docs.append(result["document"])
        return docs


class AstraDBVectorStoreTestContext(VectorStoreTestContext):
    def __init__(self, handler: "AstraDBVectorStoreHandler"):
        super().__init__()
        self.handler = handler
        self.test_id = "test_id" + random_string()

    def new_langchain_vector_store(self, **kwargs) -> EnhancedLangChainVectorStore:
        logging.info(
            f"Creating langchain vector store, implementation {self.handler.implementation}, collection {self.handler.collection_name}"
        )

        if self.handler.implementation == VectorStoreImplementation.CASSANDRA:
            vector_store = EnhancedCassandraLangChainVectorStore(
                session=None,
                keyspace="default_keyspace",
                table_name=self.handler.collection_name,
                **kwargs,
            )
        else:
            vector_store = EnhancedAstraDBLangChainVectorStore(
                collection_name=self.handler.collection_name,
                token=self.handler.token,
                api_endpoint=self.handler.api_endpoint,
                **kwargs,
            )
        logging.info("Created vector store")
        return vector_store

    def new_langchain_chat_memory(self, **kwargs) -> BaseChatMessageHistory:
        if self.handler.implementation == VectorStoreImplementation.CASSANDRA:
            return CassandraChatMessageHistory(
                session_id=self.test_id,
                session=None,
                keyspace="default_keyspace",
                table_name=self.handler.collection_name + "_chat_memory",
                **kwargs,
            )
        else:
            return AstraDBChatMessageHistory(
                session_id=self.test_id,
                token=self.handler.token,
                api_endpoint=self.handler.api_endpoint,
                collection_name=self.handler.collection_name + "_chat_memory",
                **kwargs,
            )

    def new_llamaindex_vector_store(self, **kwargs) -> EnhancedLlamaIndexVectorStore:
        logging.info(
            f"Creating llama index vector store, implementation {self.handler.implementation}, collection {self.handler.collection_name}"
        )
        if self.handler.implementation == VectorStoreImplementation.CASSANDRA:
            vector_store = EnhancedCassandraLlamaIndexVectorStore(
                session=None,
                keyspace="default_keyspace",
                table=self.handler.collection_name,
                **kwargs,
            )
        else:
            vector_store = EnhancedAstraDBLlamaIndexVectorStore(
                token=self.handler.token,
                api_endpoint=self.handler.api_endpoint,
                collection_name=self.handler.collection_name,
                **kwargs,
            )
        logging.info("Created vector store")
        return vector_store


def try_delete_with_backoff(collection: str, sleep=1, max_tries=5):
    try:
        response = AstraDBVectorStoreHandler.default_astra_client.delete_collection(
            collection
        )
        logging.info(f"delete collection {collection} response: {str(response)}")
    except Exception as e:
        max_tries -= 1
        if max_tries < 0:
            raise e

        logging.warning(f"An exception occurred deleting collection {collection}: {e}")
        time.sleep(sleep)
        try_delete_with_backoff(collection, sleep * 2, max_tries)


class AstraDBVectorStoreHandler(VectorStoreHandler):
    token = ""
    api_endpoint = ""
    env = ""
    database_id = ""
    default_astra_client = None
    delete_collection_handler = None

    @classmethod
    def initialize(cls):
        if not cls.token:
            cls.token = get_required_env("ASTRA_DB_TOKEN")
            cls.api_endpoint = get_required_env("ASTRA_DB_ENDPOINT")
            cls.env = os.environ.get("ASTRA_DB_ENV", "prod").lower()
            cls.database_id = get_required_env("ASTRA_DB_ID")
            cls.default_astra_client = AstraPyClient(
                api_endpoint=cls.api_endpoint, token=cls.token
            )
            cls.delete_collection_handler = DeleteCollectionHandler(
                try_delete_with_backoff
            )

    def __init__(self, implementation: VectorStoreImplementation):
        super().__init__(
            implementation,
            [VectorStoreImplementation.ASTRADB, VectorStoreImplementation.CASSANDRA],
        )
        self.__class__.initialize()
        self.collection_name = None

    @property
    def astra_ref(self) -> AstraRef:
        return AstraRef(
            token=self.__class__.token,
            api_endpoint=self.__class__.api_endpoint,
            collection=self.collection_name,
            id=self.__class__.database_id,
            env=self.__class__.env,
        )

    def ensure_astra_env_clean(self, blocking=False):
        logging.info(
            f"Ensuring astra env is clean (current deletions in progress: {self.__class__.delete_collection_handler.get_current_deletions()})"
        )
        collections = (
            self.__class__.default_astra_client.get_collections()
            .get("status")
            .get("collections")
        )
        logging.info(f"Existing collections: {collections}")
        if self.collection_name:
            logging.info(
                f"Deleting collection configured in the vector store: {self.collection_name}"
            )
            self.__class__.delete_collection_handler.run_delete(
                self.collection_name
            ).result()

        for name in collections:
            self.__class__.delete_collection_handler.run_delete(name)
        if blocking:
            self.__class__.delete_collection_handler.await_ongoing_deletions_completed()
            logging.info("Astra env cleanup completed")
        else:
            logging.info(
                f"Astra env cleanup started in background, proceeding (current deletions in progress: {self.__class__.delete_collection_handler.get_current_deletions()})"
            )

    def before_test(self) -> AstraDBVectorStoreTestContext:
        super().check_implementation()
        self.ensure_astra_env_clean(blocking=True)
        self.collection_name = "documents_" + random_string()
        logging.info("Start using collection: " + self.collection_name)

        if self.implementation == VectorStoreImplementation.CASSANDRA:
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

    def after_test(self):
        self.ensure_astra_env_clean(blocking=True)
