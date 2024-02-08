import argparse
import os
import subprocess
import sys
from datasets import load_dataset

INPUT_PATH = "data/imdb_train.csv"

TEST_CASES = [
    # "embeddings_batch1_chunk256", too slow
    "embeddings_batch1_chunk512",
    # "embeddings_batch10_chunk256",
    # "embeddings_batch10_chunk512",
    # "embeddings_batch50_chunk256",
    # "embeddings_batch50_chunk512",
    # "embeddings_batch100_chunk256",
    # "embeddings_batch100_chunk512",
]


# Custom type function to convert input string to a list of integers
def int_list(value):
    try:
        return [int(item) for item in value.strip("[]").split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Value {value} is not a valid list of integers"
        )


def get_values_for_testcase(test_case):
    if test_case.startswith("embeddings"):
        return ["nemo_microservice"]
        # return ["openai_ada002", "nvidia_nvolveqa40k"]
        # return ["openai_ada002", "nemo_microservice"]
        # return ["nvidia_nvolveqa40k", "nemo_microservice"]
    else:
        raise ValueError(f"Unknown testcase: {test_case}")


def run_suite(
    test_case: str,
    loops=1,
    report_dir=".",
    only_values_containing=None,
    threads_per_benchmark=None,
):
    if threads_per_benchmark is None:
        threads_per_benchmark: list[int] = [1]

    all_values = get_values_for_testcase(test_case)
    if only_values_containing is not None:
        for value in all_values:
            for filter_by in only_values_containing:
                if filter_by not in value:
                    all_values.remove(value)
                    break

    bechmarks_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.abspath(report_dir)

    filenames = []
    logs_file = os.path.join(args.reports_dir, "benchmarks.log")

    for value in all_values:
        for threads in threads_per_benchmark:
            filename = f"{test_case}-{value}-{threads}.json"
            abs_filename = os.path.join(report_dir, filename)
            os.path.exists(abs_filename) and os.remove(abs_filename)
            filenames.append(abs_filename)

            command = f"{sys.executable} -m pyperf command --copy-env -n 1 -l {loops} -t -o {abs_filename} -- {sys.executable} {bechmarks_dir}/testcases.py {logs_file} {test_case} {value} {threads}"
            print(
                f"Running suite: {test_case} with value: {value} and threads: {threads}"
            )
            try:
                subprocess.run(command.split(" "), text=True, check=True)
            except Exception as e:
                print(f"Error running suite: {e.args[0]}")
                if os.path.exists(logs_file):
                    with open(logs_file, "r") as f:
                        print(f.read())
                raise Exception("Error running suite")

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
    test_choices = test_choices + TEST_CASES
    parser.add_argument(
        "-t",
        "--test-case",
        choices=test_choices,
        required=True,
        help="Test case to run",
    )

    parser.add_argument(
        "-v",
        "--values",
        type=str,
        default="",
        help="Filter values to run (comma separated). e.g. to run only openai_ada002, use: openai_",
    )
    parser.add_argument(
        "-r",
        "--reports-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "reports"),
        help="Reports dir",
    )

    parser.add_argument(
        "-l",
        "--loops",
        type=int,
        default=1,
        help="Number of loops to run each benchmark. Results will be statistically computed over each loop.",
    )

    parser.add_argument(
        "-n",
        "--num_threads",
        type=int_list,
        default=[1],
        help="Number of threads (concurrent requests) per benchmark",
    )

    args = parser.parse_args()
    if not os.path.exists(args.reports_dir):
        os.makedirs(args.reports_dir)
    print(f"Reports dir: {args.reports_dir}")

    if args.test_case == "all":
        tests_to_run = TEST_CASES
    else:
        tests_to_run = filter(None, args.test_case.split(","))

    logs_file = os.path.join(args.reports_dir, "benchmarks.log")
    if os.path.exists(logs_file):
        os.remove(logs_file)
    print(f"Logs file: {logs_file}")

    # Download the dataset to use
    if not os.path.exists(INPUT_PATH):
        dataset = load_dataset("imdb", split="train")
        dataset.to_csv(INPUT_PATH, index=False)

    for test_case in tests_to_run:
        run_suite(
            test_case=test_case,
            report_dir=args.reports_dir,
            loops=args.loops,
            only_values_containing=args.values.split(","),
            threads_per_benchmark=args.num_threads,
        )
