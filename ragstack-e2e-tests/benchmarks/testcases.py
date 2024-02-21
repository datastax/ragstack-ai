import os
import sys
import logging
import time
import psutil
import threading
import subprocess
import asyncio

from langchain_community.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings
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


# Get the logger for the 'httpx' library
logger = logging.getLogger("httpx")
# Set the logging level to 'WARNING' to suppress 'INFO' and 'DEBUG' messages
logger.setLevel(logging.WARNING)


def log_cpu_usage(stop_event, interval, filename):
    with open(filename, "w") as f:
        while not stop_event.is_set():
            cpu_usage = psutil.cpu_percent(interval=interval, percpu=True)

            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - CPU Usage: {cpu_usage}%\n")
            # Flush the output to ensure it's written to the file
            f.flush()


def openai_ada002(batch_size):
    model = "text-embedding-ada-002"
    # Benchmarks are often skewed by server-side timeouts, so we set a relatively low
    # timeout that still allows for a reasonable period of time for inference.
    request_timeout = 7
    logging.info(f"Setting open ai request timeout to {request_timeout}")
    return OpenAIEmbeddings(
        model=model,
        chunk_size=batch_size,
        api_key=os.environ.get("OPEN_AI_KEY"),
        max_retries=0,
        retry_min_seconds=1,
        retry_max_seconds=1,
        request_timeout=request_timeout,
    )


def azure_openai_ada002(batch_size):
    model_and_deployment = "text-embedding-ada-002"
    request_timeout = 7

    return AzureOpenAIEmbeddings(
        model=model_and_deployment,
        deployment=model_and_deployment,
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        openai_api_type="azure",
        openai_api_version="2023-05-15",
        chunk_size=batch_size,
        max_retries=0,
        retry_min_seconds=1,
        retry_max_seconds=1,
        request_timeout=request_timeout,
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


def test_script(batch_size):

    logging.basicConfig(
        filename="benchmarks/reports/benchmarks.log",
        encoding="utf-8",
        level=logging.INFO,
    )
    atime = time.time()
    asyncio.run(aeval_nemo_embeddings(batch_size, 512, 32))
    logging.info(f"Total time: {time.time() - atime:.2f} seconds")


if __name__ == "__main__":
    logs_file = sys.argv[1]
    logging.basicConfig(filename=logs_file, encoding="utf-8", level=logging.INFO)
    logging.info("hello world")

if __name__ == "__main__2":

    cpu_suffix = "cpu_usage.csv"
    gpu_suffix = "gpu_usage.csv"

    try:
        setup_start = time.time()
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

        logging.info(f"Setup time: {time.time() - setup_start:.2f} seconds")
        eval_time = time.time()
        if embedding == "nemo_microservice":
            logging.info(
                f"Running test case: {test_name}/{embedding}/threads:{threads}"
            )
            if vector_database == "astra_db":
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
            if vector_database == "astra_db":
                vector_store = astra_db(embedding_model, collection_name)
                asyncio.run(
                    aeval_embeddings_with_vector_store_indexing(
                        vector_store, embedding, chunk_size, int(threads)
                    )
                )
                # asyncio.run(
                #     aeval_embeddings_with_astrapy(
                #         embedding_model, chunk_size, int(threads), collection_name
                #     )
                # )
            else:
                asyncio.run(aeval_embeddings(embedding_model, chunk_size, int(threads)))

        logging.info(f"Evaluation time: {time.time() - eval_time:.2f} seconds")

        teardown_time = time.time()
        logging.info("Test case completed successfully")

        # Terminate GPU monitor
        nvidia_smi_process.terminate()
        nvidia_smi_process.wait()

        # Terminate CPU monitor
        stop_cpu_log_event.set()
        cpu_logging_thread.join()
        logging.info(f"Teardown time: {time.time() - teardown_time:.2f} seconds")
        logging.info(f"Total time: {time.time() - setup_start:.2f} seconds")
    except Exception as e:
        logging.exception("Exception in test case")
        logging.exception(e)
        raise
