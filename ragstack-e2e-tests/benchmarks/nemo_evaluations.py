import logging
import json
import time
import asyncio
import httpx

from concurrent.futures import ThreadPoolExecutor
from astrapy_utils import astore_embeddings, store_embeddings
from utils.text_splitter import read_and_split, read_and_split_nemo

# Define NeMo microservice API request headers
HEADERS = {"accept": "application/json", "Content-Type": "application/json"}
HOSTNAME = "0.0.0.0"
SERVICE_PORT = 8080
MODEL_ID = "NV-Embed-QA"
INPUT_TYPE = "passage"  # or query


# Size of batches for bulk insertions:
#   (Langchain currently uses 20 at the time of writing)
ASTRA_DB_BATCH_SIZE = 20


async def before_request(request):
    request.extensions["start_time"] = time.time()


async def after_request(response):
    elapsed = time.time() - response.request.extensions["start_time"]
    logging.info(f"Request to {response.url} took {elapsed} seconds")


def _embed_nemo(batch_size, chunks, threads):
    import requests
    from requests.adapters import HTTPAdapter

    url = f"http://{HOSTNAME}:{SERVICE_PORT}/v1/embeddings"
    session = requests.Session()
    session.headers.update(HEADERS)

    adapter = HTTPAdapter(pool_connections=threads, pool_maxsize=threads)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    def _process_batch(batch):
        data = {
            "input": batch,
            "model": MODEL_ID,
            "input_type": "query",
        }
        response = session.post(url, data=json.dumps(data))

        if response.status_code != 200:
            logging.error(
                f"Request failed with status code {response.status_code}: {response.text}"
            )

    num_batches = len(chunks) // batch_size + (1 if len(chunks) % batch_size else 0)
    logging.info(
        f"Processing batches of size: {batch_size}, for total num_batches: {num_batches}, across {threads} connections"
    )

    inference_start = time.time()
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

    logging.getLogger("metrics").info(
        f"Inference: {time.time() - inference_start:.3f} seconds"
    )
    session.close()


async def _aembed_nemo(batch_size, chunks, threads):
    """
    TODO: Seeing significantly slower performance with httpx. Investigate.
    """
    timeout = httpx.Timeout(30.0, pool=None)
    limits = httpx.Limits(max_connections=threads, max_keepalive_connections=threads)
    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        event_hooks={"request": [before_request], "response": [after_request]},
    ) as client:
        url = f"http://{HOSTNAME}:{SERVICE_PORT}/v1/embeddings"

        async def _process_batch(batch):
            data = {
                "input": batch,
                "model": MODEL_ID,
                "input_type": INPUT_TYPE,
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
            f"Processing batches of size: {batch_size}, for total num_batches: {num_batches}, across {threads} connections"
        )

        batches = [
            chunks[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)
        ]

        inference_start = time.time()
        await asyncio.gather(*(_process_batch(batch) for batch in batches))
        logging.getLogger("metrics").info(
            f"Inference: {time.time() - inference_start:.3f} seconds"
        )


async def _aembed_nemo_and_store(batch_size, chunks, threads, collection_name):
    """
    NOTE!!!!!!!!!!: Currently experiencing server disconnects when using httpx.
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
            f"Processing batches of size: {batch_size}, for total num_batches: {num_batches}, across {threads} connections"
        )

        inference_start = time.time()
        batches = [
            chunks[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)
        ]

        await asyncio.gather(*(_process_batch(batch) for batch in batches))

        logging.getLogger("metrics").info(
            f"Inference + Indexing: {time.time() - inference_start:.3f} seconds"
        )


def _embed_nemo_and_store(batch_size, chunks, threads, collection_name):
    import requests
    from requests.adapters import HTTPAdapter

    url = f"http://{HOSTNAME}:{SERVICE_PORT}/v1/embeddings"
    session = requests.Session()
    session.headers.update(HEADERS)

    adapter = HTTPAdapter(pool_connections=threads, pool_maxsize=threads)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    def _process_batch(batch):
        data = {
            "input": batch,
            "model": MODEL_ID,
            "input_type": "query",
        }
        response = session.post(url, headers=HEADERS, data=json.dumps(data))

        if response.status_code != 200:
            logging.error(
                f"Request failed with status code {response.status_code}: {response.text}"
            )
        embeddings = [item["embedding"] for item in response.json()["data"]]
        embeddings = store_embeddings(
            batch, embeddings, threads, ASTRA_DB_BATCH_SIZE, collection_name
        )

    num_batches = len(chunks) // batch_size + (1 if len(chunks) % batch_size else 0)
    logging.info(
        f"Processing batches of size: {batch_size}, for total num_batches: {num_batches}, across {threads} connections"
    )

    inference_start = time.time()
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

    logging.getLogger("metrics").info(
        f"Inference + Indexing: {time.time() - inference_start:.3f} seconds"
    )


async def aeval_nemo_embeddings(batch_size, chunk_size, threads):
    # chunks = read_and_split(chunk_size, "text-embedding-ada-002")
    chunks = read_and_split_nemo(chunk_size)
    _embed_nemo(batch_size, chunks, threads)
    # await _aembed_nemo(batch_size, chunks, threads)


async def aeval_nemo_embeddings_with_astrapy_indexing(
    batch_size, chunk_size, threads, collection_name
):
    chunks = read_and_split_nemo(chunk_size)
    # chunks = read_and_split(chunk_size, "text-embedding-ada-002")
    _embed_nemo_and_store(batch_size, chunks, threads, collection_name)
    # await _aembed_nemo_and_store(batch_size, chunks, threads, collection_name)
