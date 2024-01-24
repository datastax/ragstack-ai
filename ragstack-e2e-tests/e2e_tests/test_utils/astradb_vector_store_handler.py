import concurrent
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import List, Callable

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


class DeleteCollectionHandler:
    def __init__(self, delete_function: Callable, max_workers=5):
        self.delete_function = delete_function
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers
        self.semaphore = threading.Semaphore(max_workers)

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
            self._run_and_release(collection),
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


class AstraDBVectorStoreHandler(VectorStoreHandler):
    def __init__(self):
        super().__init__(
            [VectorStoreImplementation.ASTRADB, VectorStoreImplementation.CASSANDRA]
        )
        env = os.environ.get("ASTRA_DB_ENV", "prod").lower()
        self.astra_ref = AstraRef(
            token=get_required_env("ASTRA_DB_TOKEN"),
            api_endpoint=get_required_env("ASTRA_DB_ENDPOINT"),
            collection="documents_" + random_string(),
            id=get_required_env("ASTRA_DB_ID"),
            env=env,
        )
        self.default_astra_client = AstraPyClient(
            api_endpoint=self.astra_ref.api_endpoint, token=self.astra_ref.token
        )
        self.delete_collection_handler = DeleteCollectionHandler(
            self.try_delete_with_backoff
        )

    def try_delete_with_backoff(self, collection: str, sleep=1, max_tries=5):
        try:
            self.default_astra_client.delete_collection(collection)
        except Exception as e:
            max_tries -= 1
            if max_tries < 0:
                raise e

            logging.warning(
                f"An exception occurred deleting collection {collection}: {e}"
            )
            time.sleep(sleep)
            self.try_delete_with_backoff(collection, sleep * 2, max_tries)

    def ensure_astra_env_clean(self, blocking=False):
        logging.info("Ensuring astra env is clean")
        self.delete_collection_handler.run_delete(self.astra_ref.collection)
        collections = (
            self.default_astra_client.get_collections().get("status").get("collections")
        )
        logging.info(f"Existing collections: {collections}")
        for name in collections:
            if name == self.astra_ref.collection:
                continue
            self.delete_collection_handler.run_delete(name)
        if blocking:
            self.delete_collection_handler.await_ongoing_deletions_completed()

    def before_test(
        self, implementation: VectorStoreImplementation
    ) -> VectorStoreTestContext:
        super().before_test(implementation)
        self.ensure_astra_env_clean(blocking=True)
        self.astra_ref.collection = "documents_" + random_string()

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
        self.ensure_astra_env_clean(blocking=False)


class EnhancedCassandraLangChainVectorStore(EnhancedLangChainVectorStore, AstraDB):
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


class EnhancedCassandraLlamaIndexVectorStore(
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
    def __init__(self, handler: AstraDBVectorStoreHandler):
        super().__init__()
        self.handler = handler
        self.test_id = "test_id" + random_string()

    def new_langchain_vector_store(self, **kwargs) -> LangChainVectorStore:
        return EnhancedCassandraLangChainVectorStore(
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
        return EnhancedCassandraLlamaIndexVectorStore(
            token=self.handler.astra_ref.token,
            api_endpoint=self.handler.astra_ref.api_endpoint,
            collection_name=self.handler.astra_ref.collection + "_chat_memory",
            **kwargs,
        )
