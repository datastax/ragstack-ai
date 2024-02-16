import os
import logging
import time
import asyncio

from langchain_community.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings

from benchmark_utils import split


async def _aembed(embeddings: Embeddings, chunks: list[str], threads: int):
    """Embeds chunks using the given embeddings model."""

    async def process_batch(batch):
        try:
            await embeddings.aembed_documents(batch)
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

    inference_start = time.time()
    logging.info(f"Inference+Store Start: {inference_start}")

    await asyncio.gather(*(process_batch(batch) for batch in batches))

    inference_end = time.time()
    logging.info(f"Inference+Store End: {inference_end}")


async def _aeval_embeddings_with_openai_client(chunk_size, threads):
    from openai import AsyncOpenAI

    chunks = split(chunk_size)

    client = AsyncOpenAI(api_key=os.environ.get("OPEN_AI_KEY"))
    model = "text-embedding-ada-002"
    await client.embeddings.create(input=chunks, model=model)


async def aeval_embeddings(embedding_model, chunk_size, threads):
    chunks = split(chunk_size)
    await _aembed(embedding_model, chunks, threads)


async def aeval_embeddings_with_vector_store(vector_store, chunk_size, threads):
    chunks = split(chunk_size)
    await _aembed_and_store(vector_store, chunks, threads)
