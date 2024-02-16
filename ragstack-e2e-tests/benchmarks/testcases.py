import os
import sys
import logging
import requests
import time
import psutil
import threading
import subprocess
import asyncio

from requests.adapters import HTTPAdapter

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.astradb import AstraDB
from langchain_core.embeddings import Embeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from nemo_evaluations import (
    aeval_nemo_embeddings,
    aeval_nemo_embeddings_with_astrapy_indexing,
)
from evaluations import (
    aeval_embeddings,
    aeval_embeddings_with_vector_store_indexing,
    aeval_embeddings_with_astrapy,
)


thread_local = threading.local()

# Get the logger for the 'httpx' library
logger = logging.getLogger("httpx")
# Set the logging level to 'WARNING' to suppress 'INFO' and 'DEBUG' messages
logger.setLevel(logging.INFO)


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


def astra_db(embeddings: Embeddings, collection_name: str) -> AstraDB:
    astra_start = time.time()
    db = AstraDB(
        embedding=embeddings,
        collection_name=collection_name,
        token=os.environ.get("ASTRA_DB_APPLICATION_TOKEN"),
        api_endpoint=os.environ.get("ASTRA_DB_API_ENDPOINT"),
    )
    astra_setup = time.time() - astra_start
    logging.info(f"AstraDB setup time: {astra_setup:.2f} seconds")
    return db


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
        collection_name = sys.argv[8]

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
            if vector_database != "none":
                asyncio.run(
                    aeval_nemo_embeddings_with_astrapy_indexing(
                        batch_size, chunk_size, int(threads), collection_name
                    )
                )
            else:
                asyncio.run(aeval_nemo_embeddings(batch_size, chunk_size, int(threads)))
        else:
            logging.info(
                f"Running test case: {test_name}/{embedding}/threads:{threads}"
            )
            embedding_model = eval(f"{embedding}({batch_size})")
            if vector_database != "none":
                # TODO: you could pass embedding and batch size and eval inside astradb
                # vector_store = eval(f"{vector_database}({embedding_model})")
                vector_store = astra_db(embedding_model, collection_name)
                asyncio.run(
                    aeval_embeddings_with_vector_store_indexing(
                        vector_store, chunk_size, int(threads)
                    )
                )
                # asyncio.run(
                #     aeval_embeddings_with_astrapy(
                #         embedding_model, chunk_size, int(threads), collection_name
                #     )
                # )
            else:
                asyncio.run(aeval_embeddings(embedding_model, chunk_size, int(threads)))

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
