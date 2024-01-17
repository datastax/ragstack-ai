import argparse
import os
import subprocess
import sys
import time

import pyperf


def get_values_for_testcase(test_case):
    if test_case.startswith("embeddings"):
        return ["openai_ada002", "nvidia_nvolveqa40k"]
    else:
        raise ValueError(f"Unknown testcase: {test_case}")


def run_suite(test_case: str, only_values_containing=[], loops=1, processes=1, report_dir="./reports"):
    all_values = get_values_for_testcase(test_case)
    if only_values_containing:
        for value in all_values:
            if only_values_containing not in value:
                all_values.remove(value)

    bechmarks_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.abspath(report_dir)

    filenames = []

    for value in all_values:
        filename = f"{test_case}-{value}.json"
        abs_filename = os.path.join(report_dir, filename)
        os.path.exists(abs_filename) and os.remove(abs_filename)
        filenames.append(abs_filename)

        command = f"{sys.executable} -m pyperf command --copy-env -p {processes} -n 1 -l {loops} -t -o {abs_filename} --verbose -- {sys.executable} {bechmarks_dir}/testcases.py {test_case} {value}"
        print(f"Running suite: {test_case} with value: {value}")
        subprocess.run(
            command.split(" "),
            text=True, check=True).check_returncode()

    if len(filenames) <= 1:
        print("Not enough files to compare")
    else:
        print("Showing comparison:")
        filenames_str = " ".join(filenames)

        comparison_command = f"{sys.executable} -m pyperf compare_to {filenames_str} --table -v -G"
        subprocess.run(
            comparison_command.split(" "),
            text=True, check=True)
    print("Done")


def run_suite_all(test_case, only_values_containing=[]):
    run_suite(test_case, only_values_containing, loops=4, processes=4)


TEST_CASES = [
    "embeddings_single_doc_256",
    "embeddings_single_doc_512",
    "embeddings_10_docs_256",
    "embeddings_10_docs_512",
    "embeddings_50_docs_256",
    "embeddings_50_docs_512",
    "embeddings_100_docs_256",
    "embeddings_100_docs_512",
]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Benchmarks runner',
        description='Run benchmarks to compare different providers and combinations')

    test_choices = ["*"]
    test_choices = test_choices + TEST_CASES
    parser.add_argument('-t', '--test-case', choices=test_choices, required=True,
                        help='Test case to run')

    parser.add_argument('-v', '--values', type=str, default="",
                        help='Filter values to run (comma separated). e.g. to run only openai_ada002, use: openai_')
    args = parser.parse_args()

    if args.test_case == "*":
        tests_to_run = TEST_CASES
    else:
        tests_to_run = filter(None, args.test_case.split(","))

    for test_case in tests_to_run:
        run_suite_all(test_case, args.values.split(","))
