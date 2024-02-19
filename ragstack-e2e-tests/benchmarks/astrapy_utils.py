from concurrent.futures import ThreadPoolExecutor
import logging
import uuid
import os
import asyncio
from itertools import islice

from typing import (
    Iterator,
    List,
    Iterable,
    Any,
    Dict,
    Coroutine,
    Tuple,
    Union,
)
from typing import Set, Union, TypeVar
from astrapy.db import AstraDB, AstraDBCollection, AsyncAstraDB, AsyncAstraDBCollection


DocDict = Dict[str, Any]  # dicts expressing entries to insert
T = TypeVar("T")
U = TypeVar("U")


def _get_documents_to_insert(
    texts: Iterable[str],
    embedding_vectors: List[List[float]],
) -> List[DocDict]:
    ids = [uuid.uuid4().hex for _ in texts]

    documents_to_insert = [
        {
            "content": b_txt,
            "_id": b_id,
            "$vector": b_emb,
        }
        for b_txt, b_emb, b_id in zip(
            texts,
            embedding_vectors,
            ids,
        )
    ]
    return documents_to_insert


def _async_collection(collection_name) -> AsyncAstraDBCollection:
    token = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
    api_endpoint = os.environ.get("ASTRA_DB_API_ENDPOINT")
    client = AsyncAstraDB(
        token=token,
        api_endpoint=api_endpoint,
        namespace="default_keyspace",
    )
    return AsyncAstraDBCollection(
        collection_name=collection_name,
        astra_db=client,
    )


def _get_collection(collection_name) -> AstraDBCollection:
    token = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
    api_endpoint = os.environ.get("ASTRA_DB_API_ENDPOINT")
    client = AstraDB(
        token=token,
        api_endpoint=api_endpoint,
        namespace="default_keyspace",
    )
    return AstraDBCollection(
        collection_name=collection_name,
        astra_db=client,
    )


async def _gated_coro(semaphore: asyncio.Semaphore, coro: Coroutine) -> Any:
    """Run a coroutine with a semaphore.
    Args:
        semaphore: The semaphore to use.
        coro: The coroutine to run.

    Returns:
        The result of the coroutine.
    """
    async with semaphore:
        return await coro


async def _gather_with_concurrency(n: Union[int, None], *coros: Coroutine) -> list:
    """Gather coroutines with a limit on the number of concurrent coroutines.

    Args:
        n: The number of coroutines to run concurrently.
        coros: The coroutines to run.

    Returns:
        The results of the coroutines.
    """
    if n is None:
        return await asyncio.gather(*coros)

    semaphore = asyncio.Semaphore(n)

    return await asyncio.gather(*(_gated_coro(semaphore, c) for c in coros))


def _batch_iterate(size: int, iterable: Iterable[T]) -> Iterator[List[T]]:
    """Utility batching function."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


async def astore_embeddings(
    texts: List[str],
    embedding_vectors: List[List[float]],
    batch_concurrency: int,
    batch_size: int,
    collection_name: str,
):
    collection = _async_collection(collection_name=collection_name)
    documents_to_insert = _get_documents_to_insert(texts, embedding_vectors)

    async def _handle_batch(document_batch: List[DocDict]) -> List[str]:
        result = await collection.insert_many(
            documents=document_batch,
            options={"ordered": False},
            partial_failures_allowed=True,
        )
        return result["status"]["insertedIds"]

    _b_max_workers = batch_concurrency
    all_ids_nested = await _gather_with_concurrency(
        _b_max_workers,
        *[
            _handle_batch(batch)
            for batch in _batch_iterate(
                batch_size,
                documents_to_insert,
            )
        ],
    )

    return [iid for id_list in all_ids_nested for iid in id_list]


def store_embeddings(
    texts: List[str],
    embedding_vectors: List[List[float]],
    batch_concurrency: int,
    batch_size: int,
    collection_name,
):
    collection = _get_collection(collection_name=collection_name)
    documents_to_insert = _get_documents_to_insert(texts, embedding_vectors)

    def _handle_batch(document_batch: List[DocDict]) -> List[str]:
        result = collection.insert_many(
            documents=document_batch,
            options={"ordered": False},
            partial_failures_allowed=False,
        )
        return result["status"]["insertedIds"]

    _b_max_workers = batch_concurrency
    with ThreadPoolExecutor(max_workers=_b_max_workers) as tpe:
        all_ids_nested = tpe.map(
            _handle_batch,
            _batch_iterate(
                batch_size,
                documents_to_insert,
            ),
        )

    return [iid for id_list in all_ids_nested for iid in id_list]
