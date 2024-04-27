import torch

from ragstack_colbert.colbert_retriever import max_similarity_torch
from ragstack_colbert.colbert_embedding_model import calculate_query_maxlen

from ragstack_colbert import ColbertEmbeddingModel
from ragstack_colbert.constant import DEFAULT_COLBERT_DIM

from torch.nn.functional import cosine_similarity


def test_max_similarity_torch():
    # Example query vector and embedding list
    query_vector = torch.tensor([1, 2, 3], dtype=torch.float32)
    embedding_list = [
        torch.tensor([2, 3, 4], dtype=torch.float32),
        torch.tensor([1, 0, 1], dtype=torch.float32),
        torch.tensor(
            [4, 5, 6], dtype=torch.float32
        ),  # This should produce the highest dot product
    ]

    # Expected result calculated manually or logically determined
    expected_max_similarity = torch.dot(
        query_vector, embedding_list[2]
    )  # Should be the highest

    # Call the function under test
    max_sim = max_similarity_torch(query_vector, embedding_list, is_cuda=False)

    # Check if the returned max similarity matches the expected value
    assert (
        max_sim.item() == expected_max_similarity.item()
    ), "The max similarity does not match the expected value."


def test_colbert_query_embeddings():
    colbert = ColbertEmbeddingModel()

    text = "who is the president of the united states?"
    queryTensor = colbert.embed_query(text)
    assert isinstance(queryTensor, torch.Tensor)
    assert queryTensor.shape == (64, 128)

    optimizer_query_tensor = colbert.optimized_query_embeddings(
        text, query_maxlen=512
    )
    assert isinstance(optimizer_query_tensor, torch.Tensor)
    assert optimizer_query_tensor.shape == (9, DEFAULT_COLBERT_DIM)

    # test query encoding
    queryEncoding = colbert.embed_query("test-query", query_maxlen=512)
    assert len(queryEncoding) == 512


def eval_optimized_query_embeddings(text: str): 
    colbert = ColbertEmbeddingModel()

    query_tensor = colbert.embed_query(text)
    assert isinstance(query_tensor, torch.Tensor)
    assert query_tensor.shape == (64, DEFAULT_COLBERT_DIM)

    optimizer_query_tensor = colbert.optimized_query_embeddings(text)
    assert isinstance(optimizer_query_tensor, torch.Tensor)

    # the dimension of the tensor
    n = optimizer_query_tensor.shape[0]
    assert DEFAULT_COLBERT_DIM == optimizer_query_tensor.shape[1]

    # evaluate the optimized query tensor's similarity with the original query tensor
    # the original query token sequence is in this order 
    # [101, 1, ..., 102, 103, ... 103]
    # so the first, second Tensor will be skipped
    tensor_adjusted = query_tensor[2:n+2]
    for i in range(len(tensor_adjusted)):
        similarity = cosine_similarity(optimizer_query_tensor[i].unsqueeze(0), tensor_adjusted[i].unsqueeze(0))
        assert similarity > 0.999

        distance = torch.norm(optimizer_query_tensor[i] - tensor_adjusted[i])
        assert abs(distance) < 0.001

    # Note: we do not evaluate 103 MASK token, all 103 tokens are embedded totally different


def test_optimized_query_embeddings():
    eval_optimized_query_embeddings("who is the president of the united states?")
    eval_optimized_query_embeddings("me?")
    eval_optimized_query_embeddings("how many planets are in the solar system?")
    eval_optimized_query_embeddings("what is the largest mammal in the world?")
    # this is 64 token long query
    eval_optimized_query_embeddings("What are the potential benefits and drawbacks of implementing a nationwide high-speed rail network in the United States, considering the diverse geographic and demographic factors across different regions, and how might these impact the overall feasibility and sustainability of such a project?")