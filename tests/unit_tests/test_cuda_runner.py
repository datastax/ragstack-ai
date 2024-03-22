from ragstack.colbert.distributed import Distributed
from ragstack.colbert.runner import sample_work_load, map_work_load
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

def test_even_distribution():
    result = sample_work_load(work_load_size=4, processors=2)
    assert all(len(chunk) == 2 for chunk in result), "All chunks should have equal size"

def test_uneven_distribution():
    result = sample_work_load(work_load_size=5, processors=2)
    assert len(result[-1]) < len(result[0]), "The last chunk should be smaller"

def test_single_processor():
    result = sample_work_load(work_load_size=5, processors=1)
    assert len(result) == 1 and len(result[0]) == 5, "Should return the entire workload as a single chunk"

def test_single_workload():
    result = sample_work_load(work_load_size=1, processors=2)
    assert len(result) == 1 and len(result[0]) == 1, "Should return the single workload as a single chunk"

def test_no_workload():
    result = sample_work_load(work_load_size=0, processors=2)
    assert len(result) == 0, "Should return an empty list"

def test_more_processors_than_workloads():
    result = sample_work_load(work_load_size=2, processors=4)
    # Expecting 2 chunks since workload cannot be split into more chunks than there are units of work
    assert len(result) == 2, "Should not create more chunks than there are workloads, even if there are more processors"

def test_even_distribution():
    collections = ["item1", "item2", "item3", "item4"]
    expected = [["item1", "item3"], ["item2", "item4"]]
    assert map_work_load(collections, 2) == expected, "Failed to evenly distribute an even number of items"

def test_uneven_distribution():
    collections = ["item1", "item2", "item3", "item4", "item5"]
    expected = [["item1", "item3", "item5"], ["item2", "item4"]]
    assert map_work_load(collections, 2) == expected, "Failed to distribute an odd number of items"

def test_single_processor():
    collections = ["item1", "item2", "item3"]
    expected = [collections]  # All items should be assigned to the single processor
    assert map_work_load(collections, 1) == expected, "Failed with a single processor"

def test_more_processors_than_collections():
    collections = ["item1", "item2"]
    # Expecting each item in its own list, assuming the function handles more processors than collections by limiting processors
    expected = [["item1"], ["item2"]]
    assert map_work_load(collections, 4) == expected, "Failed when there are more processors than collections"

def test_no_collections():
    collections = []
    expected = []  # Expecting an empty list when there are no collections
    assert map_work_load(collections, 2) == expected, "Failed with no collections"

def test_single_collection():
    collections = ["item1"]
    expected = [["item1"]]  # Single item should be in its own list
    assert map_work_load(collections, 1) == expected, "Failed with a single collection"
    assert map_work_load(collections, 2) == expected, "Failed with a single collection and more processors"

def test_fifteen_items_four_processors():
    collections = ["Item1", "Item2", "Item3", "Item4", "Item5", 
                   "Item6", "Item7", "Item8", "Item9", "Item10", 
                   "Item11", "Item12", "Item13", "Item14", "Item15"]
    expected_output = [
        ["Item1", "Item5", "Item9", "Item13"],
        ["Item2", "Item6", "Item10", "Item14"],
        ["Item3", "Item7", "Item11", "Item15"],
        ["Item4", "Item8", "Item12"]
    ]
    assert map_work_load(collections, 4) == expected_output, "The distribution of 15 items across 4 processors is incorrect"