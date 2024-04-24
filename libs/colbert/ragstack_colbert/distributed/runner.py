"""
Facilitates parallel processing of text chunk encoding by distributing workload across multiple processors or CUDA
devices. This module includes utilities to evenly distribute collections of text for encoding, leveraging multiprocessing
and CUDA capabilities for improved performance. The `Runner` class orchestrates the parallel encoding process, managing
processes and collecting their results efficiently.

Designed to optimize encoding tasks in distributed computing environments, the module ensures workload balance and
maximizes resource utilization by dynamically adjusting to the number of available processors or CUDA-enabled GPUs.
"""

import logging
from typing import Dict, List

import torch
import torch.multiprocessing as mp

from colbert.infra import ColBERTConfig

from ..objects import BaseText, EmbeddedText
from .chunk_encoder import encode_chunks
from .distributed import reconcile_nranks


def distribute_work_load(
    work_load_size: int = 1, processors: int = 1
) -> List[List[int]]:
    """
    Distributes a given workload size across a specified number of processors.

    Parameters:
        work_load_size (int): The total size of the workload to be distributed.
        processors (int): The number of processors available for workload distribution.

    Returns:
        List[List[int]]: A nested list where each sublist contains indices representing the workload assigned to each processor.
    """

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


def map_work_load(texts: List[BaseText], processors: int = 1) -> List[List[BaseText]]:
    """
    Maps a list of text chunks to a specified number of processors for distributed processing. This function
    leverages `distribute_work_load` to evenly distribute the collections among available processors.

    Parameters:
        texts (List[str]): The chunk texts to be processed.
        processors (int): The number of processors available for distribution.

    Returns:
        List[List[str]]: A nested list where each sublist contains the texts assigned to each processor.
    """

    work_loads = distribute_work_load(len(texts), processors)
    return [[texts[i] for i in workload] for workload in work_loads]


def cuda_encode_texts(
    config: ColBERTConfig,
    rank: int,
    texts: List[BaseText],
    return_dict: Dict[int, List[EmbeddedText]],
):
    """
    Encodes a collection of text chunks using CUDA-enabled devices, storing the results in a shared dictionary.
    This function is designed to be run in a separate process for each chunk of the workload.

    Parameters:
        config: The configuration settings for the encoding process.
        rank (int): The rank of the current process in the distributed setting.
        collection (List[str]): The collection of text chunks to encode.
        return_dict: A multiprocessing.Manager().dict() to store the results of the encoding process.
    """
    if torch.cuda.is_available():
        logging.info(f"encoder runs on cuda id {torch.cuda.current_device()}")
    results = encode_chunks(config=config, rank=rank, texts=texts)
    return_dict[rank] = results


class Runner:
    """
    Orchestrates the distribution and parallel processing of text chunk encoding tasks across multiple processors or CUDA (GPU)
    devices. Utilizes multiprocessing to initiate separate encoding processes and aggregates their results upon completion.

    Attributes:
        _is_cuda (bool): Indicates if CUDA is available for GPU acceleration.
        _nranks (int): The number of processor ranks determined based on availability and the provided configuration.
    """

    def __init__(self, nranks: int = 1) -> None:
        """
        Initializes the Runner with a specified number of ranks, adjusting for the availability of CUDA devices.

        Parameters:
            nranks (int): The desired number of ranks (processors) for distributing the encoding workload.
        """

        # this runner is only useful when nranks > 1
        self._is_cuda = torch.cuda.is_available()
        self._nranks = 1
        if self._is_cuda:
            self._nranks = reconcile_nranks(nranks)

    # this is the entrypoint to the distributed embedding code
    def encode(
        self,
        config: ColBERTConfig,
        texts: List[str],
        timeout: int = 60,
    ) -> List[EmbeddedText]:
        """
        Encodes a collection of text across multiple processors or CUDA devices in parallel. Manages the lifecycle
        of subprocesses, ensuring timely completion and aggregating their results.

        Parameters:
            config: The configuration settings for the encoding process.
            texts (List[str]): The text chunks to encode.
            timeout (int): The maximum time (in seconds) allowed for each subprocess to complete.

        Returns:
            A list of encoded results aggregated from all subprocesses.
        """

        manager = mp.Manager()
        return_dict = manager.dict()

        _texts = [BaseText(original_index=index, text=text) for index, text in enumerate(texts)]

        work_loads = map_work_load(_texts, self._nranks)
        logging.info(f"encoding {len(work_loads)} texts on nranks {self._nranks}")

        processes = []
        proc_info = []
        ranks = len(work_loads)
        for rank, work_load in enumerate(work_loads):
            p = mp.Process(
                target=cuda_encode_texts,
                args=(config, rank, work_load, return_dict),
            )
            p.start()
            processes.append(p)
            proc_info.append((p.pid, p.name))
            logging.debug(f"start process on rank {rank} of {self._nranks} nranks")

        timed_out_processes = []
        for p, info in zip(processes, proc_info):
            p.join(timeout=timeout)
            if p.is_alive():
                logging.error(
                    f"embedding process timed out process PID: {info[0]}, Name: {info[1]}"
                )
                timed_out_processes.append(p)
            else:
                logging.debug(
                    f"joined embedding process ID: {info[0]}, Name: {info[1]}"
                )

        if timed_out_processes:
            raise Exception(
                "one or more processes did not complete within the timeout period"
            )
        else:
            logging.info("all processes completed")

        # Aggregate results from each GPU
        result_list:List[EmbeddedText] = []
        for new_rank in range(ranks):
            result_list.extend(return_dict[new_rank])

        return result_list
