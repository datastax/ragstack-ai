import argparse
import os
import subprocess
import sys

TEST_CASES = [
    # "embeddings_batch1_chunk256", too slow
    # "embeddings_batch1_chunk512", too slow
    "embeddings_batch10_chunk256",
    "embeddings_batch10_chunk512",
    "embeddings_batch50_chunk256",
    "embeddings_batch50_chunk512",
    "embeddings_batch100_chunk256",
    "embeddings_batch100_chunk512",
]

INTENSITIES = {
    "1": {"processes": 1, "loops": 1},
    "2": {"processes": 2, "loops": 2},
    "3": {"processes": 3, "loops": 3},
    "4": {"processes": 4, "loops": 4},
    "5": {"processes": 10, "loops": 5},
}

PROCESSES = 1
LOOPS_PER_PROCESS = 1


def get_values_for_testcase(test_case):
    if test_case.startswith("embeddings"):
        return ["openai_ada002", "nvidia_nvolveqa40k"]
    else:
        raise ValueError(f"Unknown testcase: {test_case}")


def run_suite(
        test_case: str, only_values_containing=[], loops=1, processes=1, report_dir="."
):
    all_values = get_values_for_testcase(test_case)
    if only_values_containing:
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
        filename = f"{test_case}-{value}.json"
        abs_filename = os.path.join(report_dir, filename)
        os.path.exists(abs_filename) and os.remove(abs_filename)
        filenames.append(abs_filename)

        command = f"{sys.executable} -m pyperf command --copy-env -p {processes} -q -n 1 -l {loops} -t -o {abs_filename} -- {sys.executable} {bechmarks_dir}/testcases.py {logs_file} {test_case} {value}"
        print(f"Running suite: {test_case} with value: {value}")
        try:
            subprocess.run(command.split(" "), text=True, check=True).check_returncode()
        except Exception as e:
            print(f"Error running suite: {e.args[0]}")
            if os.path.exists(logs_file):
                with open(logs_file, "r") as f:
                    print(f.read())
            raise Exception(f"Error running suite")

    if len(filenames) <= 1:
        print("Not enough files to compare")
    else:
        print("Showing comparison:")
        filenames_str = " ".join(filenames)

        comparison_command = (
            f"{sys.executable} -m pyperf compare_to {filenames_str} --table -v -G"
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
        "-i",
        "--intensity",
        choices=["1", "2", "3", "4", "5"],
        default="2",
        help="Intensity of the test (1-5). The higher the number, the more iterations will be run and the more the tests will cost and take time.",
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

    for test_case in tests_to_run:
        run_suite(
            test_case=test_case,
            only_values_containing=args.values.split(","),
            report_dir=args.reports_dir,
            loops=INTENSITIES[args.intensity]["loops"],
            processes=INTENSITIES[args.intensity]["processes"],
        )
