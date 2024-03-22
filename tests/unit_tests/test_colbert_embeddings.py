from ragstack.colbert.colbert_embedding import ColbertTokenEmbeddings
from ragstack.colbert.constant import DEFAULT_COLBERT_MODEL, DEFAULT_COLBERT_DIM
import torch


def test_colbert_token_embeddings():
    colbert = ColbertTokenEmbeddings()
    assert colbert.colbert_config is not None

    passagesEmbeddings = colbert.embed_documents(["test1", "test2"])

    assert len(passagesEmbeddings) == 2

    assert passagesEmbeddings[0].text() == "test1"
    assert passagesEmbeddings[1].text() == "test2"

    # generate uuid based id
    assert passagesEmbeddings[0].doc_id() != ""
    assert passagesEmbeddings[1].doc_id() != ""

    passageEmbeddings = colbert.embed_documents(
        texts=["test1", "test2"], doc_id="test-id"
    )

    assert passageEmbeddings[0].text() == "test1"
    assert passageEmbeddings[0].doc_id() == "test-id"
    assert passageEmbeddings[1].doc_id() == "test-id"

    token_embeddings = passagesEmbeddings[0].get_all_token_embeddings()
    assert len(token_embeddings[0].get_embeddings()) == DEFAULT_COLBERT_DIM


def test_colbert_token_embeddings_with_params():
    colbert = ColbertTokenEmbeddings(
        doc_maxlen=220,
        nbits=1,
        kmeans_niters=4,
        checkpoint=DEFAULT_COLBERT_MODEL,
        query_maxlen=32,
    )
    assert colbert.colbert_config is not None

    passage_embeddings = colbert.embed_documents(["test1", "test2", "test3"])

    assert len(passage_embeddings) == 3

    assert passage_embeddings[0].text() == "test1"
    assert passage_embeddings[1].text() == "test2"

    token_embeddings = passage_embeddings[0].get_all_token_embeddings()
    assert len(token_embeddings) > 1
    assert len(token_embeddings[0].get_embeddings()) == DEFAULT_COLBERT_DIM


def test_colbert_query_embeddings():
    colbert = ColbertTokenEmbeddings()

    queryTensor = colbert.embed_query("who is the president of the united states?")
    assert isinstance(queryTensor, torch.Tensor)
    assert queryTensor.shape == (11, 128)

    # test query encoding
    queryEncoding = colbert.encode_query("test-query", query_maxlen=512)
    assert len(queryEncoding) == 512
