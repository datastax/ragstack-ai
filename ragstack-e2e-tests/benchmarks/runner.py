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


def run_suite(test_case: str, only_values=None, loops=1, processes=1, report_dir="./reports"):
    if only_values is None:
        only_values = []
    all_values = get_values_for_testcase(test_case)
    if len(only_values) > 0:
        all_values = [v for v in all_values if v in only_values]

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


def run_suite_all(test_case):
    run_suite(test_case, [], loops=4, processes=4)

if __name__ == "__main__":

    test_cases = [
        "embeddings_single_doc_256",
        "embeddings_single_doc_512",
        "embeddings_10_docs_256",
        "embeddings_10_docs_512",
        "embeddings_50_docs_256",
        "embeddings_50_docs_512",
        "embeddings_100_docs_256",
        "embeddings_100_docs_512",
    ]
    for test_case in test_cases:
        run_suite_all(test_case)

