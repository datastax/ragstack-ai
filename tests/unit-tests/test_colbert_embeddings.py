# test_embeddings.py

from ragstack.colbert.colbert_embedding import ColbertTokenEmbeddings
from ragstack.colbert.constant import DEFAULT_COLBERT_MODEL, DEFAULT_COLBERT_DIM
import torch

def test_colbert_token_embeddings():
    colbert = ColbertTokenEmbeddings()
    assert colbert.colbert_config is not None

    passagesEmbeddings = colbert.embed_documents(["test1", "test2"])

    assert len(passagesEmbeddings) == 2

    assert passagesEmbeddings[0].get_text() == "test1"
    assert passagesEmbeddings[1].get_text() == "test2"

    # generate uuid based title
    assert passagesEmbeddings[0].title() != ""
    assert passagesEmbeddings[1].title() != ""

    passageEmbeddings = colbert.embed_documents(texts=["test1", "test2"], title="test-title")

    assert passageEmbeddings[0].get_text() == "test1"
    assert passageEmbeddings[0].title() == "test-title"
    assert passageEmbeddings[1].title() == "test-title"

    tokenEmbeddings = passagesEmbeddings[0].get_all_token_embeddings()
    assert len(tokenEmbeddings[0].get_embeddings()) == DEFAULT_COLBERT_DIM


def test_colbert_token_embeddings_with_params():
    colbert = ColbertTokenEmbeddings(
        doc_maxlen=220,
        nbits=1,
        kmeans_niters=4,
        checkpoint=DEFAULT_COLBERT_MODEL,
        query_maxlen=32,
    )
    assert colbert.colbert_config is not None

    passagesEmbeddings = colbert.embed_documents(["test1", "test2", "test3"])

    assert len(passagesEmbeddings) == 3

    assert passagesEmbeddings[0].get_text() == "test1"
    assert passagesEmbeddings[1].get_text() == "test2"

    tokenEmbeddings = passagesEmbeddings[0].get_all_token_embeddings()
    assert len(tokenEmbeddings) > 1
    assert len(tokenEmbeddings[0].get_embeddings()) == DEFAULT_COLBERT_DIM

def test_colbert_query_embeddings():
    colbert = ColbertTokenEmbeddings()

    queryTensor = colbert.embed_query("who is the president of the united states?")
    assert isinstance(queryTensor, torch.Tensor)
    assert queryTensor.shape == (11, 128)


    # test query encoding
    queryEncoding = colbert.encode_query("test-query", query_maxlen=512)
    assert len(queryEncoding) == 512



