import torch
import pytest

from ragstack.colbert.cassandra_retriever import max_similarity_torch

def test_max_similarity_torch():
    # Example query vector and embedding list
    query_vector = torch.tensor([1, 2, 3], dtype=torch.float32)
    embedding_list = [
        torch.tensor([2, 3, 4], dtype=torch.float32),
        torch.tensor([1, 0, 1], dtype=torch.float32),
        torch.tensor([4, 5, 6], dtype=torch.float32)  # This should produce the highest dot product
    ]

    # Expected result calculated manually or logically determined
    expected_max_similarity = torch.dot(query_vector, embedding_list[2])  # Should be the highest

    # Call the function under test
    max_sim = max_similarity_torch(query_vector, embedding_list, is_cuda=False)

    # Check if the returned max similarity matches the expected value
    assert max_sim.item() == expected_max_similarity.item(), "The max similarity does not match the expected value."
