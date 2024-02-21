import pyperf
from testcases import test_script


def test_me(timee):
    # time.sleep(timee)
    pass


def setup():
    pass


def run_benchmark():
    # --copy-env -t -p {processes} -n {values_per_benchmark} -l {loops} -o {abs_filename} -- {sys.executable} {benchmarks_dir}/testcases.py {logs_file} {test_name} {embedding_name} {batch_size} {chunk_size} {threads} {vector_database} {collection_name}"
    runner = pyperf.Runner(values=2, processes=1, loops=2)
    # runner.parse_args(
    #     args=["--stats", "-n", "1", "-l", "1", "-p", "1", "-v", "-o", "testout.json"]
    # )
    # runner.parse_args(args=["--debug-single-value"])
    # print(f"okay: {runner.args}")

    batch_sizes = [1, 100]
    # times = 1

    for batch_size in batch_sizes:
        runner.bench_func(
            f"test_bench_test_{batch_size}",
            test_script,
            batch_sizes,
        )


if __name__ == "__main__":
    run_benchmark()
