"""
Provides utilities for setting up and managing a distributed computing environment using PyTorch. This includes
functions for determining the optimal number of ranks (processes) based on available CUDA devices, finding free
network ports, and initializing PyTorch's distributed process group. The module is designed to facilitate
distributed training or computation tasks across multiple GPUs or nodes, enhancing performance and efficiency.

Although not required for single-device setups, these utilities become crucial for scaling to multi-device or
multi-node projects, offering streamlined setup and teardown processes for distributed operations.

Note:
This module currently only distributes work on a single machine.
"""

import logging
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def reconcile_nranks(nranks: int) -> int:
    """
    Determines the appropriate number of ranks (parallel processes) for distributed operations based on the
    passed value and the available CUDA (GPU) or CPU devices.

    Parameters:
        nranks (int): The desired number of ranks (parallel processes). If less than 1, the function aims to use all available processors.

    Returns:
        int: The number of ranks to be used, which may be adjusted based on the availability of processors.
    """
    __cuda = torch.cuda.is_available()
    if __cuda:
        __cuda_device_count = torch.cuda.device_count()
        if nranks < 1:
            return __cuda_device_count
        else:
            return min(nranks, __cuda_device_count)
    else:
        if nranks < 1:
            return 1
        else:
            # currently let user set nranks on CPU
            return nranks


def find_free_port():
    """
    Finds a free network port on the localhost that can be used for inter-process communication in distributed setups.

    Returns:
        str: A string representation of the free port number.
    """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def setup_process(
    rank: int,
    master_addr: str,
    master_port: str,
    world_size: int,
    backend="nccl",
) -> None:
    """
    Initializes a distributed process group for a given rank within the world size of a distributed computing
    environment. This setup is crucial for coordinated operations across multiple processes.

    Parameters:
        rank (int): The rank of the current process within the distributed group.
        master_addr (str): The IP address of the master node for coordination.
        master_port (str): The port on the master node used for communication.
        world_size (int): The total number of processes in the distributed environment.
        backend (str, optional): The backend to use for distributed operations. Defaults to "nccl".
    """

    logging.info(
        f"setting up {rank=} {world_size=} {backend=} {master_addr=} {master_port=}"
    )

    # set up the master's ip address so this child process can coordinate
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # Initializes the default distributed process group, and this will also initialize the distributed package.
    dist.init_process_group(
        backend, rank=rank, world_size=world_size, init_method="env://"
    )
    logging.info(
        f"{rank=} init_process_group completed {world_size=} {backend=} {master_port=}"
    )
    # TODO: find out when to destroy the process group
    # dist.destroy_process_group()


class Distributed:
    """
    A singleton class designed to initialize and manage the distributed environment for PyTorch applications. It
    ensures that the distributed setup is only initialized once and provides access to the total number of ranks
    or processes involved in the computation.

    The class automatically determines the optimal configuration based on available resources and the desired
    number of ranks, setting up inter-process communication and preparing the environment for distributed
    operations.
    """

    _instance = None  # Keep instance reference
    _is_initialized = False
    _world_size = 0

    def __new__(cls, *args, **kwargs):
        """
        Ensures that only one instance of the Distributed class is created (Singleton pattern).
        This method checks if an instance already exists; if not, it creates a new one, ensuring that the
        distributed environment setup is only executed once.

        Parameters:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The singleton instance of the Distributed class.
        """

        if not cls._instance:
            cls._instance = super(Distributed, cls).__new__(cls)
        return cls._instance

    def __init__(self, nranks: int):
        """
        Initializes the Distributed class instance by setting up the distributed environment if it hasn't been
        initialized already. This setup includes determining the number of ranks and configuring the process group.

        Parameters:
            nranks (int): The desired number of ranks (processes) to use in the distributed environment. If the
                          number is less than 1, the system will attempt to use all available CUDA devices.
        """
        if not self._is_initialized:
            self._setup(nranks)
            self._is_initialized = True

    def _setup(self, nranks: int):
        """
        Configures the distributed environment by determining the optimal number of ranks and initializing the
        process group for distributed operations. This method is intended to be called internally once during the
        initialization of the Distributed class.

        Parameters:
            nranks (int): The desired number of ranks (processes) for the distributed environment. This value is
                          used to calculate the effective world size and to initialize the process group accordingly.
        """

        master_addr = "127.0.0.1"
        master_port = find_free_port()
        if nranks < 1:
            nranks = torch.cuda.device_count()
        world_size = min(torch.cuda.device_count(), nranks)
        logging.info(f"setting up resource group {world_size}")
        self._world_size = world_size
        mp.spawn(
            setup_process,
            args=(
                master_addr,
                master_port,
                world_size,
            ),
            nprocs=world_size,
        )
        logging.info(f"resource group setup completed {self._world_size}")

    def world_size(self):
        """
        Retrieves the world size (the total number of processes participating in the distributed environment)
        that was determined during the setup.

        Returns:
            int: The world size of the distributed environment.
        """
        return self._world_size
