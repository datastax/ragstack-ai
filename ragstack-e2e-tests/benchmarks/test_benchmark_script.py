import pyperf
from testcases import test_script
import time


def test_me(timee):
    time.sleep(timee)


def setup():
    pass


def run_benchmark():
    # --copy-env -t -p {processes} -n {values_per_benchmark} -l {loops} -o {abs_filename} -- {sys.executable} {benchmarks_dir}/testcases.py {logs_file} {test_name} {embedding_name} {batch_size} {chunk_size} {threads} {vector_database} {collection_name}"
    runner = pyperf.Runner(values=1, processes=1, loops=1)
    runner.parse_args(
        args=["--copy-env", "-w 1", "-p 1", "-l 1", "-n 1", "-o testout.json"]
    )
    batch_sizes = [1]
    times = 1

    for batch_size in batch_sizes:
        runner.bench_func(f"test_bench_test_{batch_size}", lambda: test_me(times))


if __name__ == "__main__":
    run_benchmark()
