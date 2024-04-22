import torch

from ragstack_colbert.colbert_embedding import ChunkData, ColbertEmbedding
from ragstack_colbert.constant import DEFAULT_COLBERT_DIM, DEFAULT_COLBERT_MODEL


def test_colbert_token_embeddings():
    colbert = ColbertEmbedding()
    assert colbert.colbert_config is not None

    chunks = [
        ChunkData(text="test1"),
        ChunkData(text="test2"),
    ]

    embedded_chunks = colbert.embed_chunks(chunks=chunks)

    assert len(embedded_chunks) == 2

    assert embedded_chunks[0].data.text == "test1"
    assert embedded_chunks[1].data.text == "test2"

    # generate uuid based id
    assert embedded_chunks[0].doc_id != ""
    assert embedded_chunks[1].doc_id != ""

    embedded_chunks = colbert.embed_chunks(chunks=chunks, doc_id="test-id")

    assert embedded_chunks[0].data.text == "test1"
    assert embedded_chunks[0].doc_id == "test-id"
    assert embedded_chunks[1].doc_id == "test-id"

    embeddings = embedded_chunks[0].embeddings
    assert len(embeddings[0]) == DEFAULT_COLBERT_DIM


def test_colbert_token_embeddings_with_params():
    colbert = ColbertEmbedding(
        doc_maxlen=220,
        nbits=1,
        kmeans_niters=4,
        checkpoint=DEFAULT_COLBERT_MODEL,
        query_maxlen=32,
    )
    assert colbert.colbert_config is not None

    chunks = [
        ChunkData(text="test1"),
        ChunkData(text="test2"),
        ChunkData(text="test3"),
    ]

    embedded_chunks = colbert.embed_chunks(chunks=chunks)

    assert len(embedded_chunks) == 3

    assert embedded_chunks[0].data.text == "test1"
    assert embedded_chunks[1].data.text == "test2"

    embeddings = embedded_chunks[0].embeddings
    assert len(embeddings) > 1
    assert len(embeddings[0]) == DEFAULT_COLBERT_DIM


def test_colbert_query_embeddings():
    colbert = ColbertEmbedding()

    queryTensor = colbert.embed_query("who is the president of the united states?")
    assert isinstance(queryTensor, torch.Tensor)
    assert queryTensor.shape == (11, 128)

    # test query encoding
    queryEncoding = colbert.encode_query("test-query", query_maxlen=512)
    assert len(queryEncoding) == 512
