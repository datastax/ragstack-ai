import os
import logging
import time
import asyncio

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from benchmark_utils import read_and_split
from astrapy_utils import astore_embeddings


async def _aembed(embeddings: Embeddings, chunks: list[str], threads: int):
    """Embeds chunks using the given embeddings model."""

    async def process_batch(batch) -> list[list[float]]:
        try:
            return await embeddings.aembed_documents(batch)
        except Exception as e:
            logging.error(f"Failed to embed chunk: {e}")

    batch_size = len(chunks) // threads + (1 if len(chunks) % threads else 0)
    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
    logging.info(
        f"Splitting chunks into {len(batches)} batches of size {batch_size} for {threads} threads"
    )

    inference_start = time.time()
    logging.info(f"Inference Start: {inference_start}")

    await asyncio.gather(*(process_batch(batch) for batch in batches))

    inference_end = time.time()
    logging.info(f"Inference End: {inference_end}")


async def _aembed_and_store(vector_store: VectorStore, chunks: list[str], threads: int):
    """Embeds and stores chunks into the vector store."""

    async def process_batch(batch):
        try:
            logging.info(f"Storing batch of size: {len(batch)}")
            await vector_store.aadd_texts(batch)
        except Exception as e:
            logging.error(f"Failed to embed chunk: {e}")

    batch_size = len(chunks) // threads + (1 if len(chunks) % threads else 0)
    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
    logging.info(
        f"Splitting chunks into {len(batches)} batches of size {batch_size} for {threads} threads"
    )

    start_time = time.time()
    await asyncio.gather(*(process_batch(batch) for batch in batches))
    logging.info(f"Total Inference + Indexing Time: {time.time() - start_time}")


async def _aeval_embeddings_and_store_with_astrapy(
    embeddings: Embeddings, chunks: list[str], threads: int, collection_name: str
):
    """
    This uses the OpenAIEmbeddings class to create vectors, then opts to use AstraPy
    directly to store them instead of LangChain's AstraDBVectorStore.
    """

    async def process_batch(batch) -> list[list[float]]:
        try:
            return await embeddings.aembed_documents(batch)
        except Exception as e:
            logging.error(f"Failed to embed chunk: {e}")

    batch_size = len(chunks) // threads + (1 if len(chunks) % threads else 0)
    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
    logging.info(
        f"Splitting chunks into {len(batches)} batches of size {batch_size} for {threads} threads"
    )

    inference_start = time.time()
    embeddings = await asyncio.gather(*(process_batch(batch) for batch in batches))
    logging.info(f"Total Inference Time: {time.time() - inference_start}")

    # Flatten embeddings
    embeddings = [item for sublist in embeddings for item in sublist]

    indexing_start = time.time()
    await astore_embeddings(chunks, embeddings, threads, batch_size, collection_name)
    logging.info(f"Total Indexing Time: {time.time() - indexing_start}")


async def aeval_embeddings(embedding_model, chunk_size, threads):
    chunks = read_and_split(chunk_size)
    await _aembed(embedding_model, chunks, threads)


async def aeval_embeddings_with_vector_store_indexing(
    vector_store, chunk_size, threads
):
    chunks = read_and_split(chunk_size)
    await _aembed_and_store(vector_store, chunks, threads)


async def aeval_embeddings_with_astrapy(
    embedding_model, chunk_size, threads, collection_name
):
    chunks = read_and_split(chunk_size)
    await _aeval_embeddings_and_store_with_astrapy(
        embedding_model, chunks, threads, collection_name
    )
