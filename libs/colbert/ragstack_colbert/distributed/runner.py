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

from ..objects import TextChunk, TextEmbedding
from .chunk_encoder import ChunkEncoder


def reconcile_nranks(nranks: int) -> int:
    """
    Determines the appropriate number of ranks (parallel processes) for distributed operations based on the
    passed value and the available CUDA (GPU) or CPU devices.

    Parameters:
        nranks (int): The desired number of ranks (parallel processes). If less than 1, the function aims to use all available processors.

    Returns:
        int: The number of ranks to be used, which may be adjusted based on the availability of processors.
    """
    cuda = torch.cuda.is_available()
    if cuda:
        cuda_device_count = torch.cuda.device_count()
        if nranks < 1:
            return cuda_device_count
        else:
            return min(nranks, cuda_device_count)
    else:
        if nranks < 1:
            return 1
        else:
            # currently let user set nranks on CPU
            return nranks


def map_work_load(chunks: List[TextChunk], processors: int = 1) -> List[List[TextChunk]]:
    """
    Maps a list of text chunks to a specified number of processors for distributed processing. This function
    leverages `distribute_work_load` to evenly distribute the collections among available processors.

    Parameters:
        chunks (List[TextChunk]): The chunk texts to be processed.
        processors (int): The number of processors available for distribution.

    Returns:
        List[List[TextChunk]]: A nested list where each sublist contains the text chunks assigned to each processor.
    """

    work_load_size = len(chunks)
    if work_load_size == 0:
        return []

    # ensure no empty workload assigns to a processor
    processors = min(work_load_size, processors)
    # Initialize an empty list for each processor
    work_loads = [[] for _ in range(processors)]

    for index, chunk in enumerate(chunks):
        work_loads[index % processors].append(chunk)

    return work_loads


def cuda_encode_chunks(
    config: ColBERTConfig,
    rank: int,
    chunks: List[TextChunk],
    return_dict: Dict[int, List[TextChunk]],
):
    """
    Encodes a collection of text chunks using CUDA-enabled devices, storing the results in a shared dictionary.
    This function is designed to be run in a separate process for each chunk of the workload.

    Parameters:
        config: The configuration settings for the encoding process.
        rank (int): The rank of the current process in the distributed setting.
        chunks (List[TextChunk]): The text chunks to encode.
        return_dict: A multiprocessing.Manager().dict() to store the results of the encoding process.
    """
    if torch.cuda.is_available():
        logging.info(f"encoder runs on cuda id {torch.cuda.current_device()}")
    encoder = ChunkEncoder(config=config)
    return_dict[rank] = encoder.encode_chunks(chunks=chunks)


class Runner:
    """
    Orchestrates the distribution and parallel processing of text chunk encoding tasks across multiple processors or CUDA (GPU)
    devices. Utilizes multiprocessing to initiate separate encoding processes and aggregates their results upon completion.

    Attributes:
        _is_cuda (bool): Indicates if CUDA is available for GPU acceleration.
        _nranks (int): The number of processor ranks determined based on availability and the provided configuration.
    """

    def __init__(self, config: ColBERTConfig, nranks: int = 1) -> None:
        """
        Initializes the Runner with a specified number of ranks, adjusting for the availability of CUDA devices.

        Parameters:
            nranks (int): The desired number of ranks (processors) for distributing the encoding workload.
        """

        # this runner is only useful when nranks > 1
        self._is_cuda = torch.cuda.is_available()
        self._colbert_config = config
        self._nranks = 1
        if self._is_cuda:
            self._nranks = reconcile_nranks(nranks)

    # this is the entrypoint to the distributed embedding code
    def encode(
        self,
        chunks: List[TextChunk],
        timeout: int = 60,
    ) -> List[TextEmbedding]:
        """
        Encodes a collection of text chunks across multiple processors or CUDA devices in parallel. Manages the lifecycle
        of subprocesses, ensuring timely completion and aggregating their results.

        Parameters:
            config: The configuration settings for the encoding process.
            chunks (List[TextChunk]): The text chunks to encode.
            timeout (int): The maximum time (in seconds) allowed for each subprocess to complete.

        Returns:
            A list of TextEmbedding results aggregated from all subprocesses. Order is not guaranteed.
        """

        manager = mp.Manager()
        return_dict = manager.dict()

        work_loads = map_work_load(chunks, self._nranks)
        logging.info(f"encoding {len(work_loads)} texts on nranks {self._nranks}")

        processes = []
        proc_info = []
        ranks = len(work_loads)
        for rank, work_load in enumerate(work_loads):
            p = mp.Process(
                target=cuda_encode_chunks,
                args=(self._colbert_config, rank, work_load, return_dict),
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
        result_list:List[TextEmbedding] = []
        for new_rank in range(ranks):
            result_list.extend(return_dict[new_rank])

        return result_list
