import torch
from ragstack_colbert import ColbertEmbeddingModel
from ragstack_colbert.constant import DEFAULT_COLBERT_DIM, DEFAULT_COLBERT_MODEL


def test_colbert_token_embeddings():
    colbert = ColbertEmbeddingModel()

    texts = ["test1", "test2"]

    embeddings = colbert.embed_texts(texts=texts)

    assert len(embeddings) == 2
    assert len(embeddings[0][0]) == DEFAULT_COLBERT_DIM


def test_colbert_token_embeddings_with_params():
    colbert = ColbertEmbeddingModel(
        doc_maxlen=220,
        nbits=2,
        kmeans_niters=4,
        checkpoint=DEFAULT_COLBERT_MODEL,
        query_maxlen=32,
    )

    texts = ["test1", "test2", "text3"]

    embeddings = colbert.embed_texts(texts=texts)

    assert len(embeddings) == 3

    assert len(embeddings[0][0]) == DEFAULT_COLBERT_DIM


def test_colbert_query_embeddings():
    colbert = ColbertEmbeddingModel()

    embedding = colbert.embed_query("who is the president of the united states?")
    query_tensor = torch.tensor(embedding)
    assert query_tensor.shape == (12, 128)

    # test query encoding
    embedding = colbert.embed_query("test-query", query_maxlen=512)
    assert len(embedding) == 512
