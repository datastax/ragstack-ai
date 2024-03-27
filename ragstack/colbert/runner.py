import logging
import torch
import torch.multiprocessing as mp
from typing import List

from .distributed import reconcile_nranks

"""
Sample the work load across n number of processors
generate a list to distribute the work load across the processors
"""


def sample_work_load(work_load_size: int = 1, processors: int = 1):
    if work_load_size == 0:
        return []
    # ensure no empty workload assigns to a processor
    processors = min(work_load_size, processors)
    # Initialize an empty list for each processor
    result = [[] for _ in range(processors)]

    # Distribute each workload to a processor in a round-robin fashion
    for i in range(work_load_size):
        result[i % processors].append(i)
    return result


"""
Map collections work load to processors
"""


def map_work_load(collections: List[str], processors: int = 1) -> List[List[str]]:
    work_loads = sample_work_load(len(collections), processors)
    return [[collections[i] for i in workload] for workload in work_loads]


"""
This class runs process on CUDA devices in multi-process mode.
"""


class Runner:
    def __init__(self, func, nranks: int = 1):
        self.func = func
        # nrank is the processor number of ranks
        # this runner is only useful when nrank > 1
        self._cuda = torch.cuda.is_available()
        self._nranks = reconcile_nranks(nranks)

    def run(self, encoder, collections: List[str], title: str, *args, **kwargs):
        manager = mp.Manager()
        results = manager.list()

        work_load_size = len(collections)
        work_loads = sample_work_load(work_load_size, self.nranks)

        processes = []
        gpu_id = 0
        for work_load in work_loads:
            p = mp.Process(
                target=self.func, args=(gpu_id, encoder, work_load, title, results)
            )
            processes.append(p)
            gpu_id = gpu_id + 1
            p.start()

        for p in processes:
            p.join()

        logging.info(f"running {self.nranks} ranks")

        return list(results)
