from ragstack.colbert.distributed import Distributed
import torch

def test_distributed():
    distributed = Distributed(nranks=-1)
    assert distributed.world_size() == 0

    distributed = Distributed(nranks=8)
    assert distributed.world_size() == 0


def test_distributed_with_nranks():
    # this is a singletone so this would not work.
    # we need to test on cuda initialization
    if torch.cuda.is_available() == False:
        distributed = Distributed(nranks=8)
        assert distributed.world_size() == 0
    else:
        world_size = torch.cuda.device_count()
        distributed = Distributed(nranks=world_size)
        assert distributed.world_size() == world_size