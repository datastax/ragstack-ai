import torch

from ragstack.colbert.colbert_embedding import ColbertTokenEmbeddings
from ragstack.colbert.constant import (DEFAULT_COLBERT_DIM,
                                       DEFAULT_COLBERT_MODEL)


def test_colbert_token_embeddings():
    colbert = ColbertTokenEmbeddings()
    assert colbert.colbert_config is not None

    embedded_chunks = colbert.embed_chunks(["test1", "test2"])

    assert len(embedded_chunks) == 2

    assert embedded_chunks[0].text() == "test1"
    assert embedded_chunks[1].text() == "test2"

    # generate uuid based id
    assert embedded_chunks[0].doc_id() != ""
    assert embedded_chunks[1].doc_id() != ""

    embedded_chunks = colbert.embed_chunks(
        texts=["test1", "test2"], doc_id="test-id"
    )

    assert embedded_chunks[0].text() == "test1"
    assert embedded_chunks[0].doc_id() == "test-id"
    assert embedded_chunks[1].doc_id() == "test-id"

    vectors = embedded_chunks[0].vectors()
    assert len(vectors[0]) == DEFAULT_COLBERT_DIM


def test_colbert_token_embeddings_with_params():
    colbert = ColbertTokenEmbeddings(
        doc_maxlen=220,
        nbits=1,
        kmeans_niters=4,
        checkpoint=DEFAULT_COLBERT_MODEL,
        query_maxlen=32,
    )
    assert colbert.colbert_config is not None

    embedded_chunks = colbert.embed_chunks(["test1", "test2", "test3"])

    assert len(embedded_chunks) == 3

    assert embedded_chunks[0].text() == "test1"
    assert embedded_chunks[1].text() == "test2"

    vectors = embedded_chunks[0].vectors()
    assert len(vectors) > 1
    assert len(vectors[0]) == DEFAULT_COLBERT_DIM


def test_colbert_query_embeddings():
    colbert = ColbertTokenEmbeddings()

    queryTensor = colbert.embed_query("who is the president of the united states?")
    assert isinstance(queryTensor, torch.Tensor)
    assert queryTensor.shape == (11, 128)

    # test query encoding
    queryEncoding = colbert.encode_query("test-query", query_maxlen=512)
    assert len(queryEncoding) == 512
