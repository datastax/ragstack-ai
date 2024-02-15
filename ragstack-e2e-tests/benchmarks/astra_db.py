from itertools import islice
import uuid
import os
import asyncio

from typing import Iterator, List, Optional, Iterable, Any, Dict, Coroutine, Union
from typing import Callable, Set, Union, TypeVar

from astrapy.db import AsyncAstraDB, AsyncAstraDBCollection

DocDict = Dict[str, Any]  # dicts expressing entries to insert
T = TypeVar("T")
U = TypeVar("U")


def _unique_list(lst: List[T], key: Callable[[T], U]) -> List[T]:
    visited_keys: Set[U] = set()
    new_lst = []
    for item in lst:
        item_key = key(item)
        if item_key not in visited_keys:
            visited_keys.add(item_key)
            new_lst.append(item)
    return new_lst


def _get_documents_to_insert(
    texts: Iterable[str],
    embedding_vectors: List[List[float]],
) -> List[DocDict]:
    if ids is None:
        ids = [uuid.uuid4().hex for _ in texts]
    if metadatas is None:
        metadatas = [{} for _ in texts]

    documents_to_insert = [
        {
            "content": b_txt,
            "_id": b_id,
            "$vector": b_emb,
            "metadata": b_md,
        }
        for b_txt, b_emb, b_id, b_md in zip(
            texts,
            embedding_vectors,
            ids,
            metadatas,
        )
    ]
    # make unique by id, keeping the last
    uniqued_documents_to_insert = _unique_list(
        documents_to_insert[::-1],
        lambda document: document["_id"],
    )[::-1]
    return uniqued_documents_to_insert


def async_collection() -> AsyncAstraDBCollection:
    token = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
    api_endpoint = os.environ.get("ASTRA_DB_API_ENDPOINT")
    client = AsyncAstraDB(
        token=token,
        api_endpoint=api_endpoint,
        namespace="default-namespace",
    )
    return AsyncAstraDBCollection(
        collection_name="test",
        astra_db=client,
    )


async def gated_coro(semaphore: asyncio.Semaphore, coro: Coroutine) -> Any:
    """Run a coroutine with a semaphore.
    Args:
        semaphore: The semaphore to use.
        coro: The coroutine to run.

    Returns:
        The result of the coroutine.
    """
    async with semaphore:
        return await coro


async def gather_with_concurrency(n: Union[int, None], *coros: Coroutine) -> list:
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

    return await asyncio.gather(*(gated_coro(semaphore, c) for c in coros))


def batch_iterate(size: int, iterable: Iterable[T]) -> Iterator[List[T]]:
    """Utility batching function."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


async def aadd_embeddings(
    texts: List[str],
    embedding_vectors: List[List[float]],
    batch_concurrency: int,
    batch_size: int,
):
    collection = async_collection()
    documents_to_insert = _get_documents_to_insert(texts, embedding_vectors)

    async def _handle_batch(document_batch: List[DocDict]) -> List[str]:
        await collection.insert_many(
            documents=document_batch,
            options={"ordered": False},
            partial_failures_allowed=True,
        )

    _b_max_workers = batch_concurrency
    all_ids_nested = await gather_with_concurrency(
        _b_max_workers,
        *[
            _handle_batch(batch)
            for batch in batch_iterate(
                batch_size,
                documents_to_insert,
            )
        ],
    )

    return [iid for id_list in all_ids_nested for iid in id_list]
