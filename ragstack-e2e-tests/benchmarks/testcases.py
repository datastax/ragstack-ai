import os
import sys
import logging
import requests
import json
import time
import psutil
import concurrent
import threading
import subprocess

from requests.adapters import HTTPAdapter

from langchain.text_splitter import TokenTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from benchmarks.runner import INPUT_PATH

# Define NeMo microservice API request headers
HEADERS = {"accept": "application/json", "Content-Type": "application/json"}
HOSTNAME = "0.0.0.0"
SERVICE_PORT = 8080
MODEL_ID = "NV-Embed-QA"

thread_local = threading.local()


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


def _embed(embeddings: Embeddings, docs: list[str], threads: int):
    def process_chunk(chunk):
        try:
            logging.debug(f"Embedding {len(chunk)} documents")
            embeddings.embed_documents(chunk)
        except Exception as e:
            logging.error(f"Failed to embed chunk: {e}")

    # Determine the size of each chunk based on the desired number of threads
    chunk_size = len(docs) // threads + (1 if len(docs) % threads else 0)

    # Create chunks of documents to be processed
    chunks = [docs[i : i + chunk_size] for i in range(0, len(docs), chunk_size)]

    inference_start = time.time()
    logging.info(f"Inference Start: {inference_start}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Exception occurred in thread: {e}")

    inference_end = time.time()
    logging.info(f"Inference End: {inference_end}")


def _split(chunk_size: int) -> list[str]:
    start_split = time.time()
    READ_SIZE = 34603000
    logging.info(f"Ingesting file of bytes: {READ_SIZE}")

    with open(INPUT_PATH, "r") as file:
        input_data = file.read(READ_SIZE)

    # TODO: NeMo token limit is 512, though using anything above a chunk_size of 300 will result in
    # sporadic token length errors.
    text_splitter = TokenTextSplitter(chunk_size=max(chunk_size, 300), chunk_overlap=0)
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


def embeddings_batch1_chunk256(embeddings_fn, threads):
    docs = _split(256)
    if embeddings_fn is not None:
        _embed(embeddings_fn(1, threads), docs, threads)
    else:
        _local_nemo_embedding(1, docs, threads)


def embeddings_batch1_chunk512(embeddings_fn, threads):
    docs = _split(512)
    if embeddings_fn is not None:
        _embed(embeddings_fn(1, threads), docs, threads)
    else:
        _local_nemo_embedding(1, docs, threads)


def embeddings_batch10_chunk256(embeddings_fn, threads):
    docs = _split(256)
    if embeddings_fn is not None:
        _embed(embeddings_fn(10, threads), docs, threads)
    else:
        _local_nemo_embedding(10, docs, threads)


def embeddings_batch10_chunk512(embeddings_fn, threads):
    docs = _split(512)
    if embeddings_fn is not None:
        _embed(embeddings_fn(10, threads), docs, threads)
    else:
        _local_nemo_embedding(10, docs, threads)


def embeddings_batch50_chunk256(embeddings_fn, threads):
    docs = _split(256)
    if embeddings_fn is not None:
        _embed(embeddings_fn(50, threads), docs, threads)
    else:
        _local_nemo_embedding(50, docs, threads)


def embeddings_batch50_chunk512(embeddings_fn, threads):
    docs = _split(512)
    if embeddings_fn is not None:
        _embed(embeddings_fn(50, threads), docs, threads)
    else:
        _local_nemo_embedding(50, docs, threads)


def embeddings_batch100_chunk256(embeddings_fn, threads):
    docs = _split(256)
    if embeddings_fn is not None:
        _embed(embeddings_fn(100, threads), docs, threads)
    else:
        _local_nemo_embedding(100, docs, threads)


def embeddings_batch100_chunk512(embeddings_fn, threads):
    docs = _split(512)
    if embeddings_fn is not None:
        _embed(embeddings_fn(100, threads), docs, threads)
    else:
        _local_nemo_embedding(100, docs, threads)


def _local_nemo_embedding(batch_size, docs, threads):
    num_batches = len(docs) // batch_size + (1 if len(docs) % batch_size else 0)
    logging.info(
        f"Processing batches of size: {batch_size}, for total num_batches: {num_batches}"
    )

    def _process_batch(batch):
        session = get_session()
        data = {
            "input": batch,
            "model": MODEL_ID,
            "input_type": "query",
        }
        data_json = json.dumps(data)
        response = session.post(url, headers=HEADERS, data=data_json)

        if response.status_code != 200:
            logging.error(
                f"Request failed with status code {response.status_code}: {response.text}"
            )
        return response

    inference_start = time.time()
    logging.info(f"Inference Start: {inference_start}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        batches = [
            docs[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)
        ]
        future_to_batch = {
            executor.submit(_process_batch, batch): batch for batch in batches
        }
        for future in concurrent.futures.as_completed(future_to_batch):
            try:
                future.result()
            except Exception as exc:
                logging.error(f"Batch generated an exception: {exc}")

    inference_end = time.time()
    logging.info(f"Inference End: {inference_end}")


def openai_ada002(batch_size, threads):
    # test network latency first -- can subtract this from each call manually for now.
    total_latency = 0
    bad_embeds = OpenAIEmbeddings(chunk_size=batch_size, api_key="bad_creds")
    for _ in range(10):
        logging.info("Calling openai with bad credentials")
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
    )


def nvidia_nvolveqa40k(batch_size, threads):
    # 50 is the max supported batch size
    return NVIDIAEmbeddings(
        model="nvolveqa_40k",
        max_batch_size=min(50, batch_size),
        model_type="query",
    )


if __name__ == "__main__":
    cpu_suffix = "cpu_usage.csv"
    gpu_suffix = "gpu_usage.csv"

    try:
        logs_file = sys.argv[1]
        logging.basicConfig(filename=logs_file, encoding="utf-8", level=logging.INFO)

        test_case = sys.argv[2]
        embeddings = sys.argv[3]
        threads = sys.argv[4]

        cpu_logs_file = "-".join([test_case, embeddings, threads, cpu_suffix])
        gpu_logs_file = "-".join([test_case, embeddings, threads, gpu_suffix])
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

        # Construct the URL
        url = f"http://{HOSTNAME}:{SERVICE_PORT}/v1/embeddings"

        if embeddings == "nemo_microservice":
            logging.info(
                f"Running test case: {test_case}/{embeddings}/threads:{threads}"
            )
            embeddings = None
            eval(f"{test_case}({embeddings}, {threads})")
        else:
            logging.info(f"Running test case: {test_case}/{embeddings}")
            eval(f"{test_case}({embeddings}, {threads})")

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
