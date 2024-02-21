import argparse
import os
import random
import string
import subprocess
import sys
from datasets import load_dataset
from enum import Enum

from astrapy.db import AstraDB

from utils.text_splitter import INPUT_PATH


class TestCase(Enum):
    EMBEDDINGS_BATCH1_CHUNK512 = {
        "name": "embeddings_batch1_chunk512",
        "batch_size": 1,
        "chunk_size": 512,
    }
    # EMBEDDINGS_BATCH1_CHUNK256 = {
    #     "name": "embeddings_batch1_chunk256",
    #     "batch_size": 1,
    #     "chunk_size": 256,
    # }
    # EMBEDDINGS_BATCH10_CHUNK512 = {
    #     "name": "embeddings_batch10_chunk512",
    #     "batch_size": 10,
    #     "chunk_size": 512,
    # }
    # EMBEDDINGS_BATCH10_CHUNK256 = {
    #     "name": "embeddings_batch10_chunk256",
    #     "batch_size": 10,
    #     "chunk_size": 256,
    # }
    # EMBEDDINGS_BATCH50_CHUNK512 = {
    #     "name": "embeddings_batch50_chunk512",
    #     "batch_size": 50,
    #     "chunk_size": 512,
    # }
    # EMBEDDINGS_BATCH50_CHUNK256 = {
    #     "name": "embeddings_batch50_chunk256",
    #     "batch_size": 50,
    #     "chunk_size": 256,
    # }
    # EMBEDDINGS_BATCH100_CHUNK512 = {
    #     "name": "embeddings_batch100_chunk512",
    #     "batch_size": 100,
    #     "chunk_size": 512,
    # }
    # EMBEDDINGS_BATCH100_CHUNK256 = {
    #     "name": "embeddings_batch100_chunk256",
    #     "batch_size": 100,
    #     "chunk_size": 256,


# }


class EmbeddingModels(Enum):
    NEMO = {"name": "nemo_microservice", "dimensions": 1024}
    # OPENAI = {"name": "openai_ada002", "dimensions": 1536}
    # AZURE_OPENAI = {"name": "azure_openai_ada002", "dimensions": 1536}


# Custom type function to convert input string to a list of integers
def int_list(value):
    try:
        items = [item.strip() for item in value.strip("[]").split(",") if item.strip()]
        return [int(item) for item in items]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Value {value} is not a valid list of integers"
        )


def run_suite(
    test_case: TestCase,
    values_per_benchmark=1,
    loops=1,
    processes=1,
    report_dir=".",
    only_values_containing=None,
    threads_per_benchmark=None,
    vector_database="none",
):
    if threads_per_benchmark is None:
        threads_per_benchmark: list[int] = [1]

    embedding_models = [EmbeddingModels.value for EmbeddingModels in EmbeddingModels]
    if only_values_containing is not None:
        for embedding_model in embedding_models:
            for filter_by in only_values_containing:
                if filter_by not in embedding_model["name"]:
                    embedding_models.remove(embedding_model)
                    break

    benchmarks_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.abspath(report_dir)

    filenames = []
    logs_file = os.path.join(args.reports_dir, "benchmarks.log")

    for embedding_model in embedding_models:
        # Models should produce the same embedding dimensions, so create one collection here
        # and reuse it across for tests that reuse the model.
        if vector_database == "astra_db":
            astra = AstraDB(
                token=os.environ.get("ASTRA_DB_APPLICATION_TOKEN"),
                api_endpoint=os.environ.get("ASTRA_DB_API_ENDPOINT"),
                namespace="default_keyspace",
            )
            collection_name = "".join(random.choices(string.ascii_letters, k=10))
            collection = astra.create_collection(
                collection_name=collection_name,
                dimension=embedding_model["dimensions"],
                metric="cosine",
            )
            print("Created astra_db collection: ", collection.collection_name)
        else:
            collection_name = ""

        for threads in threads_per_benchmark:
            test_name = test_case["name"]
            embedding_name = embedding_model["name"]

            filename = f"{test_name}-{embedding_name}-{threads}.json"
            abs_filename = os.path.join(report_dir, filename)
            os.path.exists(abs_filename) and os.remove(abs_filename)
            filenames.append(abs_filename)

            batch_size = test_case["batch_size"]
            chunk_size = test_case["chunk_size"]
            # command = f"{sys.executable} -m pyperf command --copy-env --stats -p {processes} -n {values_per_benchmark} -l {loops} -o {abs_filename} -- {sys.executable} {benchmarks_dir}/testcases.py {logs_file} {test_name} {embedding_name} {batch_size} {chunk_size} {threads} {vector_database} {collection_name}"
            command = f"{sys.executable} -m pyperf command --copy-env --stats -p {processes} -n {values_per_benchmark} -l {loops} -o {abs_filename} -- {sys.executable} {benchmarks_dir}/test.py {logs_file} {test_name} {embedding_name} {batch_size} {chunk_size} {threads} {vector_database} {collection_name}"
            print(
                f"Running suite: {test_name} with model: {embedding_model} and threads: {threads}"
            )
            if vector_database == "astra_db":
                print(f"Indexing embeddings into {vector_database}/{collection_name}")

            try:
                subprocess.run(command.split(" "), text=True, check=True)
            except Exception as e:
                print(f"Error running suite: {e.args[0]}")
                if os.path.exists(logs_file):
                    with open(logs_file, "r") as f:
                        print(f.read())

                if vector_database == "astra_db":
                    print("Deleting astra_db collection: ", collection.collection_name)
                    astra.delete_collection(collection.collection_name)

                raise Exception("Error running suite")

        if vector_database == "astra_db":
            print("Deleting astra_db collection: ", collection.collection_name)
            astra.delete_collection(collection.collection_name)

    if len(filenames) <= 1:
        print("Not enough files to compare")
    else:
        filenames_str = " ".join(filenames)
        print("Showing comparison between files: {filenames_str}")

        comparison_command = (
            f"{sys.executable} -m pyperf compare_to --table -v {filenames_str}"
        )
        subprocess.run(comparison_command.split(" "), text=True, check=True)
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Benchmarks runner",
        description="Run benchmarks to compare different providers and combinations",
    )

    test_choices = ["all"]
    test_choices = test_choices + [t.value["name"] for t in TestCase]
    parser.add_argument(
        "-t",
        "--test-case",
        choices=test_choices,
        required=True,
        help="Test case to run",
    )

    parser.add_argument(
        "-m",
        "--models",
        type=str,
        default="",
        help="Filter models to run (comma separated). e.g. to run only openai_ada002, use: openai_",
    )
    parser.add_argument(
        "-r",
        "--reports-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "reports"),
        help="Reports dir",
    )

    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=1,
        help="The number of independent processes to run each benchmark. These run sequentially by default, and thus do not affect CPU/GPU access. Running multiple processes ensures sources of randomness (hash collisions, ASLR) do not affect results.",
    )

    parser.add_argument(
        "-l",
        "--loops",
        type=int,
        default=1,
        help="Number of loops a benchmark is executed within a single run. Helps to stabilize a single measurement.",
    )

    parser.add_argument(
        "-n",
        "--values",
        type=int,
        default=1,
        help="Number of values to run each benchmark. Results will be statistically computed over each value. A higher number here improves statistical robustness.",
    )

    parser.add_argument(
        "--threads",
        type=int_list,
        default=[1],
        help="Number of threads (concurrent requests) per benchmark",
    )

    parser.add_argument(
        "--vector-database",
        type=str,
        choices=["none", "astra_db"],
        default="none",
        help="Selects the vector database for storing embeddings. "
        "'none': No database used."
        "'astra_db': Stores embeddings in Astra DB. Requires setting "
        "environment variables `ASTRA_DB_APPLICATION_TOKEN` and "
        "`ASTRA_DB_API_ENDPOINT`.",
    )

    args = parser.parse_args()
    if not os.path.exists(args.reports_dir):
        os.makedirs(args.reports_dir)
    print(f"Reports dir: {args.reports_dir}")

    if args.test_case == "all":
        tests_to_run = [t.value for t in TestCase]
    else:
        test_names_to_run = filter(None, args.test_case.split(","))
        tests_to_run = [
            test_case
            for name in test_names_to_run
            for test_case in TestCase
            if test_case.value["name"] == name
        ]

    logs_file = os.path.join(args.reports_dir, "benchmarks.log")
    if os.path.exists(logs_file):
        os.remove(logs_file)
    print(f"Logs file: {logs_file}")

    # Download the dataset to use
    if not os.path.exists(INPUT_PATH):
        directory = os.path.dirname(INPUT_PATH)
        if not os.path.exists(directory):
            os.makedirs(directory)

        dataset = load_dataset("imdb", split="train")
        dataset.to_csv(INPUT_PATH, index=False)
    print("Using dataset: ", INPUT_PATH)

    for test_case in tests_to_run:
        run_suite(
            test_case=test_case,
            report_dir=args.reports_dir,
            values_per_benchmark=args.values,
            loops=args.loops,
            processes=args.processes,
            only_values_containing=args.models.split(","),
            threads_per_benchmark=args.threads,
            vector_database=args.vector_database,
        )
