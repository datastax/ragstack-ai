import os
import logging
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

"""
Initialize the torch distributed environment.
Distributed only sets up once
"""

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def setup_process(
        rank: int,
        master_addr: str,
        master_port: str,
        world_size: int,
        backend = "nccl",
    ) -> None:
    logging.info(f"setting up {rank=} {world_size=} {backend=} {master_addr=} {master_port=}")

    # set up the master's ip address so this child process can coordinate
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    # Initializes the default distributed process group, and this will also initialize the distributed package.
    dist.init_process_group(backend, rank=rank, world_size=world_size, init_method="env://")
    logging.info(f"{rank=} init_process_group completed {world_size=} {backend=} {master_port=}")
    # TODO: find out when to destroy the process group
    # dist.destroy_process_group()

class Distributed:
    _instance = None  # Keep instance reference
    _is_initialized = False
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Distributed, cls).__new__(cls)
        return cls._instance

    def __init__(self, nranks: int):
        if not self._is_initialized:
            self._setup(nranks)
            self._is_initialized = True

    def _setup(self, nranks: int):
        master_addr = "127.0.0.1"
        master_port = find_free_port()
        world_size = min(torch.cuda.device_count(), nranks)
        logging.info(f"setting up resource group {world_size=}")
        mp.spawn(setup_process, args=(master_addr,master_port,world_size,), nprocs=world_size)


