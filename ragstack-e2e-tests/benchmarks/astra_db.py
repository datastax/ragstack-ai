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


def async_collection(collection_name) -> AsyncAstraDBCollection:
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


def get_collection(collection_name) -> AstraDBCollection:
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


def _get_missing_from_batch(
    document_batch: List[DocDict], insert_result: Dict[str, Any]
) -> Tuple[List[str], List[DocDict]]:
    if "status" not in insert_result:
        raise ValueError(
            f"API Exception while running bulk insertion: {str(insert_result)}"
        )
    batch_inserted = insert_result["status"]["insertedIds"]
    # estimation of the preexisting documents that failed
    missed_inserted_ids = {document["_id"] for document in document_batch} - set(
        batch_inserted
    )
    errors = insert_result.get("errors", [])
    # careful for other sources of error other than "doc already exists"
    num_errors = len(errors)
    unexpected_errors = any(
        error.get("errorCode") != "DOCUMENT_ALREADY_EXISTS" for error in errors
    )
    if num_errors != len(missed_inserted_ids) or unexpected_errors:
        raise ValueError(f"API Exception while running bulk insertion: {str(errors)}")
    # deal with the missing insertions as upserts
    missing_from_batch = [
        document
        for document in document_batch
        if document["_id"] in missed_inserted_ids
    ]
    return batch_inserted, missing_from_batch


async def astore_embeddings(
    texts: List[str],
    embedding_vectors: List[List[float]],
    batch_concurrency: int,
    batch_size: int,
    collection_name: str,
):
    # TODO: PAss collection name in here
    collection = async_collection(collection_name=collection_name)
    documents_to_insert = _get_documents_to_insert(texts, embedding_vectors)

    async def _handle_batch(document_batch: List[DocDict]) -> List[str]:
        logging.info("Inserting many: ", len(document_batch))
        im_result = await collection.insert_many(
            documents=document_batch,
            options={"ordered": False},
            partial_failures_allowed=True,
        )

        batch_inserted, missing_from_batch = _get_missing_from_batch(
            document_batch, im_result
        )

        if len(missing_from_batch) > 0:
            logging.warn(
                "Some documents were not inserted, trying to replace them. This may skew results"
            )

        async def _handle_missing_document(missing_document: DocDict) -> str:
            replacement_result = await collection.find_one_and_replace(
                filter={"_id": missing_document["_id"]},
                replacement=missing_document,
            )
            return replacement_result["data"]["document"]["_id"]

        batch_replaced = await gather_with_concurrency(
            batch_concurrency,
            *[_handle_missing_document(doc) for doc in missing_from_batch],
        )
        return batch_inserted + batch_replaced

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


def store_embeddings(
    texts: List[str],
    embedding_vectors: List[List[float]],
    batch_concurrency: int,
    batch_size: int,
    collection_name,
):
    collection = get_collection(collection_name=collection_name)
    documents_to_insert = _get_documents_to_insert(texts, embedding_vectors)

    def _handle_batch(document_batch: List[DocDict]) -> List[str]:
        logging.info("Inserting many: ", len(document_batch))
        im_result = collection.insert_many(
            documents=document_batch,
            options={"ordered": False},
            partial_failures_allowed=True,
        )

        batch_inserted, missing_from_batch = _get_missing_from_batch(
            document_batch, im_result
        )

        if len(missing_from_batch) > 0:
            logging.warn(
                "Some documents were not inserted, trying to replace them. This may skew results"
            )

        def _handle_missing_document(missing_document: DocDict) -> str:
            replacement_result = self.collection.find_one_and_replace(  # type: ignore[union-attr]
                filter={"_id": missing_document["_id"]},
                replacement=missing_document,
            )
            return replacement_result["data"]["document"]["_id"]

        _u_max_workers = batch_concurrency
        with ThreadPoolExecutor(max_workers=_u_max_workers) as tpe2:
            batch_replaced = list(
                tpe2.map(
                    _handle_missing_document,
                    missing_from_batch,
                )
            )
        return batch_inserted + batch_replaced

    _b_max_workers = batch_concurrency
    with ThreadPoolExecutor(max_workers=_b_max_workers) as tpe:
        all_ids_nested = tpe.map(
            _handle_batch,
            batch_iterate(
                batch_size,
                documents_to_insert,
            ),
        )

    return [iid for id_list in all_ids_nested for iid in id_list]
