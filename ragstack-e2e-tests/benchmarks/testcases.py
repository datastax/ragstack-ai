import os
import sys
import logging
import requests
import json
import time
import psutil
import threading
import subprocess
import asyncio
import httpx

from requests.adapters import HTTPAdapter

from langchain.text_splitter import TokenTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.astradb import AstraDB
from langchain_community.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from runner import INPUT_PATH, ASTRA_DB_BATCH_SIZE
from astra_db import aadd_embeddings


# Define NeMo microservice API request headers
HEADERS = {"accept": "application/json", "Content-Type": "application/json"}
HOSTNAME = "0.0.0.0"
SERVICE_PORT = 8081
MODEL_ID = "NV-Embed-QA"

# The number of chars to read of the input file. A smaller value here will
# result in faster benchmarks, but may affect accuracy if not enough chunks
# are created.
#
# The default file downloaded is 33MB.
CHARS_TO_READ = 5000000

thread_local = threading.local()


# Get the logger for the 'httpx' library
logger = logging.getLogger("httpx")
# Set the logging level to 'WARNING' to suppress 'INFO' and 'DEBUG' messages
logger.setLevel(logging.DEBUG)


def get_session():
    """
    Get a session for the current thread, creating it if needed.

    This should allow for more efficient usage of connections in highly
    multi-threaded environments.
    """
    if not hasattr(thread_local, "session"):
        # Initialize a new session for each thread
        thread_local.session = requests.Session()
        # Increase the pool size to allow for more concurrent connections
        adapter = HTTPAdapter(pool_maxsize=64)
        thread_local.session.mount("http://", adapter)
        thread_local.session.mount("https://", adapter)

    return thread_local.session


def log_cpu_usage(stop_event, interval, filename):
    with open(filename, "w") as f:
        while not stop_event.is_set():
            cpu_usage = psutil.cpu_percent(interval=interval, percpu=True)

            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - CPU Usage: {cpu_usage}%\n")
            # Flush the output to ensure it's written to the file
            f.flush()


def _split(chunk_size: int) -> list[str]:
    start_split = time.time()

    logging.info(f"Reading {CHARS_TO_READ} characters from {INPUT_PATH}")
    with open(INPUT_PATH, "r") as file:
        input_data = file.read(CHARS_TO_READ)

    # TODO: NeMo token limit is 512, though using anything above a chunk_size of 300 will result in
    # sporadic token length errors.
    text_splitter = TokenTextSplitter(chunk_size=min(chunk_size, 300), chunk_overlap=0)
    split_texts = text_splitter.split_text(input_data)
    docs = []
    for split in split_texts:
        docs.append(split)

    average_length = sum(len(doc) for doc in docs) / len(docs) if docs else 0
    logging.info(
        f"Created number of documents: {len(docs)} with avg chunk size: {average_length:.2f}"
    )
    end_split = time.time()
    split_time = end_split - start_split
    logging.info(f"Text split time: {split_time:.2f} seconds")
    return docs


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


async def _aembed_nemo(batch_size, chunks, threads):
    timeout = httpx.Timeout(20.0)
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


async def _aembed_nemo_and_store(batch_size, chunks, threads):
    timeout = httpx.Timeout(20.0)
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

            # TODO: Store the embeddings in the vector store

            logging.info(f"Response: {response}")
            ids = await aadd_embeddings(batch, response, threads, ASTRA_DB_BATCH_SIZE)
            logging.info(f"IDS: {ids}")

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


def openai_ada002(batch_size):
    model = "text-embedding-ada-002"
    # test network latency first -- can subtract this from each call manually for now.
    total_latency = 0
    bad_embeds = OpenAIEmbeddings(
        model=model, chunk_size=batch_size, api_key="bad_creds"
    )
    logging.info("Calling openai with bad credentials")
    for _ in range(10):
        start_time = time.time()
        try:
            bad_embeds.embed_documents(["expect unauthorized error"])
        except Exception as e:
            end_time = time.time()
            network_latency = end_time - start_time
            total_latency += network_latency

    if total_latency == 0:
        logging.error("expected openai requests to fail and log network latency")
        raise Exception("failed to get network latency from OpenAI")

    average_latency = total_latency / 10
    logging.info(f"OpenAI Average Network Latency (s): {average_latency}")

    return OpenAIEmbeddings(
        chunk_size=batch_size,
        api_key=os.environ.get("OPEN_AI_KEY"),
        max_retries=0,  # ensure client doesn't retry requests and skew results. If this fails, we want to see it
        retry_min_seconds=0,
        retry_max_seconds=1,
    )


def nvidia_nvolveqa40k(batch_size):
    # 50 is the max supported batch size
    return NVIDIAEmbeddings(
        model="nvolveqa_40k",
        max_batch_size=min(50, batch_size),
        model_type="query",
    )


def astra_db(embeddings) -> AstraDB:
    return AstraDB(
        embedding=embeddings,
        collection_name="test_collection",
        token=os.environ.get("ASTRA_DB_APPLICATION_TOKEN"),
        api_endpoint=os.environ.get("ASTRA_DB_API_ENDPOINT"),
    )


async def _aeval_nemo_embeddings(batch_size, chunk_size, threads):
    chunks = _split(chunk_size)
    await _aembed_nemo(batch_size, chunks, threads)


async def _aeval_nemo_embeddings_with_vector_store(batch_size, chunk_size, threads):
    chunks = _split(chunk_size)
    await _aembed_nemo_and_store(batch_size, chunks, threads)


async def _aeval_embeddings(embedding_model, chunk_size, threads, vector_store):
    docs = _split(chunk_size)
    await _aembed(embedding_model, docs, threads)


async def _aeval_embeddings_with_vector_store(vector_store, chunk_size, threads):
    docs = _split(chunk_size)
    await _aembed_and_store(vector_store, docs, threads)


if __name__ == "__main__":
    cpu_suffix = "cpu_usage.csv"
    gpu_suffix = "gpu_usage.csv"

    try:
        logs_file = sys.argv[1]
        logging.basicConfig(filename=logs_file, encoding="utf-8", level=logging.INFO)

        test_name = sys.argv[2]
        embedding = sys.argv[3]
        batch_size = int(sys.argv[4])
        chunk_size = int(sys.argv[5])
        threads = sys.argv[6]
        vector_database = sys.argv[7]

        cpu_logs_file = "-".join([test_name, embedding, threads, cpu_suffix])
        gpu_logs_file = "-".join([test_name, embedding, threads, gpu_suffix])
        cpu_logs_file = f"benchmarks/reports/{cpu_logs_file}"
        gpu_logs_file = f"benchmarks/reports/{gpu_logs_file}"

        logging.info(f"CPU logs file: {cpu_logs_file}")
        logging.info(f"GPU logs file: {gpu_logs_file}")

        # Begin CPU usage monitor
        stop_cpu_log_event = threading.Event()
        cpu_logging_thread = threading.Thread(
            target=log_cpu_usage,
            args=(stop_cpu_log_event, 0.25, cpu_logs_file),
        )
        cpu_logging_thread.start()

        # Begin GPU usage monitor
        nvidia_smi_cmd = [
            "nvidia-smi",
            "--query-gpu=index,timestamp,utilization.gpu,utilization.memory",
            "--format=csv",
            "-lms",
            "250",
            ">",
            gpu_logs_file,
        ]
        nvidia_smi_process = subprocess.Popen(" ".join(nvidia_smi_cmd), shell=True)

        if embedding == "nemo_microservice":
            logging.info(
                f"Running test case: {test_name}/{embedding}/threads:{threads}"
            )
            if vector_database is not "none":
                asyncio.run(
                    _aeval_nemo_embeddings_with_vector_store(
                        batch_size, chunk_size, int(threads)
                    )
                )
            else:
                asyncio.run(
                    _aeval_nemo_embeddings(batch_size, chunk_size, int(threads))
                )
        else:
            logging.info(
                f"Running test case: {test_name}/{embedding}/threads:{threads}"
            )
            embedding_model = eval(f"{embedding}({batch_size})")
            if vector_database is not "none":
                vector_store = eval(f"{vector_database}({embedding_model})")
                asyncio.run(
                    _aeval_embeddings_with_vector_store(
                        vector_store, chunk_size, int(threads)
                    )
                )
            else:
                asyncio.run(
                    _aeval_embeddings(embedding_model, chunk_size, int(threads))
                )

        logging.info("Test case completed successfully")

        # Terminate GPU monitor
        nvidia_smi_process.terminate()
        nvidia_smi_process.wait()

        # Terminate CPU monitor
        stop_cpu_log_event.set()
        cpu_logging_thread.join()
    except Exception as e:
        logging.exception("Exception in test case")
        logging.exception(e)
        raise
