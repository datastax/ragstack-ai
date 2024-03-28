import logging
import torch
import torch.multiprocessing as mp
from typing import List

from .distributed import reconcile_nranks
from .passage_encoder import encode_passages

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


def cuda_encode_passages(config, rank: int, collection, doc_id, return_dict):
    results = encode_passages(config, rank, collection, doc_id)
    return_dict[rank] = results
    device_id = torch.cuda.current_device()
    logging.info("encoder runs on cuda id {device_id}")


"""
This class runs process on CUDA devices in multi-process mode.
"""


class Runner:
    def __init__(self, nranks: int = 1):
        # nrank is the processor number of ranks
        # this runner is only useful when nrank > 1
        self._is_cuda = torch.cuda.is_available()
        self._nranks = 1
        if self._is_cuda:
            self._nranks = reconcile_nranks(nranks)

    def encode(self, config, collections: List[str], doc_id: str):
        manager = mp.Manager()
        return_dict = manager.dict()

        work_loads = map_work_load(collections, self._nranks)
        logging.info(f"work loads runs on {len(work_loads)} gpu nranks {self._nranks}")

        processes = []
        rank = 0
        for work_load in work_loads:
            p = mp.Process(
                target=cuda_encode_passages, args=(config, rank, work_load, doc_id, return_dict)
            )
            processes.append(p)
            rank = rank + 1
            p.start()
            logging.info(f"start process on rank {rank} nranks {self._nranks}")

        for p in processes:
            p.join()

        # Aggregate results from each GPU
        result_list = []
        for new_rank in range(rank):
            result_list.extend(return_dict[new_rank])

        return result_list

