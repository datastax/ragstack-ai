import pyperf
from benchmarks.testcases import test_script


def setup():
    pass


def run_benchmark():
    runner = pyperf.Runner(processes=1, loops=1, values=1)
    batch_sizes = [1, 100]

    for batch_size in batch_sizes:
        runner.bench_func(
            f"test_bench_test_{batch_size}", lambda: test_script(batch_size)
        )


if __name__ == "__main__":
    run_benchmark()
