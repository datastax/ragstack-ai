import torch
import pytest
from typing import List

from ragstack.colbert.cassandra_retriever import max_similarity_torch
from ragstack.colbert.colbert_embedding import calculate_query_maxlen

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


def test_maxlen_less_than_min_num():
    tokens = [["word1"], ["word2", "word3"]]
    min_num = 5
    max_num = 10
    assert calculate_query_maxlen(tokens, min_num, max_num) == min_num

def test_maxlen_between_min_and_max_num():
    tokens = [["word1", "word2", "word3"], ["word1", "word2"]]
    min_num = 2
    max_num = 8
    # The longest list has 3 tokens. The expected result is the next power of 2 greater than 3, but less than max_num.
    assert calculate_query_maxlen(tokens, min_num, max_num) == 4

def test_maxlen_greater_than_max_num():
    tokens = [["word"] * 11]  # 11 tokens in the longest list
    min_num = 5
    max_num = 10
    assert calculate_query_maxlen(tokens, min_num, max_num) == max_num

# Additional test: Exact power of two within min and max
def test_maxlen_exact_power_of_two():
    tokens = [["word"] * 4]  # Exact power of

