import logging
import json
import time
import asyncio
import httpx

from concurrent.futures import ThreadPoolExecutor
from runner import ASTRA_DB_BATCH_SIZE
from benchmark_utils import read_and_split
from astra_db import astore_embeddings, store_embeddings

# Define NeMo microservice API request headers
HEADERS = {"accept": "application/json", "Content-Type": "application/json"}
HOSTNAME = "0.0.0.0"
SERVICE_PORT = 8081
MODEL_ID = "NV-Embed-QA"


async def _aembed_nemo(batch_size, chunks, threads):
    timeout = httpx.Timeout(30.0, pool=None)
    limits = httpx.Limits(max_connections=threads, max_keepalive_connections=threads)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        url = f"http://{HOSTNAME}:{SERVICE_PORT}/v1/embeddings"

        async def _process_batch(batch):
            data = {
                "input": batch,
                "model": MODEL_ID,
                "input_type": "query",
            }
            data_json = json.dumps(data)
            response = await client.post(url, headers=HEADERS, data=data_json)

            if response.status_code != 200:
                logging.error(
                    f"Request failed with status code {response.status_code}: {response.text}"
                )
            return response

        num_batches = len(chunks) // batch_size + (1 if len(chunks) % batch_size else 0)
        logging.info(
            f"Processing batches of size: {batch_size}, for total num_batches: {num_batches}"
        )

        inference_start = time.time()
        logging.info(f"Inference Start: {inference_start}")

        batches = [
            chunks[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)
        ]

        await asyncio.gather(*(_process_batch(batch) for batch in batches))

        inference_end = time.time()
        logging.info(f"Inference End: {inference_end}")


async def _aembed_nemo_and_store(batch_size, chunks, threads, collection_name):
    """
    NOTE: Currently experiencing server disconnects when using httpx.
    """
    logging.info("Embedding nemo and storing")
    # timeout = httpx.Timeout(30.0, pool=None)
    timeout = httpx.Timeout(None)
    limits = httpx.Limits(
        max_connections=threads, max_keepalive_connections=10, keepalive_expiry=None
    )
    async with httpx.AsyncClient(timeout=timeout, limits=limits, http2=True) as client:
        url = f"http://{HOSTNAME}:{SERVICE_PORT}/v1/embeddings"

        async def _process_batch(batch):
            data = {
                "input": batch,
                "model": MODEL_ID,
                "input_type": "query",
            }
            data_json = json.dumps(data)
            response = await client.post(url, headers=HEADERS, data=data_json)

            if response.status_code != 200:
                logging.error(
                    f"Request failed with status code {response.status_code}: {response.text}"
                )
            response = response.json()
            embeddings = [item["embedding"] for item in response["data"]]

            await astore_embeddings(
                batch, embeddings, threads, ASTRA_DB_BATCH_SIZE, collection_name
            )

        num_batches = len(chunks) // batch_size + (1 if len(chunks) % batch_size else 0)
        logging.info(
            f"Processing batches of size: {batch_size}, for total num_batches: {num_batches}"
        )

        inference_start = time.time()
        logging.info(f"Inference Start: {inference_start}")

        batches = [
            chunks[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)
        ]

        await asyncio.gather(*(_process_batch(batch) for batch in batches))

        inference_end = time.time()
        logging.info(f"Inference End: {inference_end}")


def _embed_nemo_and_store(batch_size, chunks, threads, collection_name):
    import requests

    logging.info("Synchronously Embedding nemo and storing")
    url = f"http://{HOSTNAME}:{SERVICE_PORT}/v1/embeddings"

    def _process_batch(batch):
        data = {
            "input": batch,
            "model": MODEL_ID,
            "input_type": "query",
        }
        response = requests.post(url, headers=HEADERS, data=json.dumps(data))

        if response.status_code != 200:
            logging.error(
                f"Request failed with status code {response.status_code}: {response.text}"
            )
        embeddings = [item["embedding"] for item in response.json()["data"]]
        embeddings = store_embeddings(
            batch, embeddings, threads, ASTRA_DB_BATCH_SIZE, collection_name
        )
        logging.info(f"Stored embeddings: {embeddings}")

    num_batches = len(chunks) // batch_size + (1 if len(chunks) % batch_size else 0)
    logging.info(
        f"Processing batches of size: {batch_size}, for total num_batches: {num_batches}"
    )

    inference_start = time.time()
    logging.info(f"Inference Start: {inference_start}")

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(_process_batch, batch)
            for batch in [
                chunks[i * batch_size : (i + 1) * batch_size]
                for i in range(num_batches)
            ]
        ]
        for future in futures:
            future.result()  # Wait for all futures to complete

    inference_end = time.time()
    logging.info(f"Inference End: {inference_end}")


async def aeval_nemo_embeddings(batch_size, chunk_size, threads):
    chunks = read_and_split(chunk_size)
    await _aembed_nemo(batch_size, chunks, threads)


async def aeval_nemo_embeddings_with_vector_store(
    batch_size, chunk_size, threads, collection_name
):
    chunks = read_and_split(chunk_size)
    # await _aembed_nemo_and_store(batch_size, chunks, threads, collection_name)
    _embed_nemo_and_store(batch_size, chunks, threads, collection_name)
