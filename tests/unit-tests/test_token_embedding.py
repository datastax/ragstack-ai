import pytest
from ragstack.colbert.token_embedding import PerTokenEmbeddings, PassageEmbeddings, DEFAULT_COLBERT_DIM, DEFAULT_COLBERT_MODEL

# Test PerTokenEmbeddings Class
@pytest.fixture
def per_token_embeddings_fixture():
    return PerTokenEmbeddings(id=1, part=0, title="Sample Title")

@pytest.fixture
def embeddings_list():
    return [0.1, 0.2, 0.3]

def test_per_token_embeddings_initialization(per_token_embeddings_fixture):
    assert per_token_embeddings_fixture.id == 1
    assert per_token_embeddings_fixture.part == 0
    assert per_token_embeddings_fixture.title == "Sample Title"

def test_add_and_get_embeddings(per_token_embeddings_fixture, embeddings_list):
    per_token_embeddings_fixture.add_embeddings(embeddings_list)
    assert per_token_embeddings_fixture.get_embeddings() == embeddings_list

# Test PassageEmbeddings Class
@pytest.fixture
def passage_embeddings_fixture():
    return PassageEmbeddings(text="This is a sample passage.", title="Sample Passage Title")

@pytest.fixture
def token_embeddings():
    te = PerTokenEmbeddings(id=2, part=1, title="Token Title")
    te.add_embeddings([0.4, 0.5, 0.6])
    return te

def test_passage_embeddings_initialization(passage_embeddings_fixture):
    assert passage_embeddings_fixture.get_text() == "This is a sample passage."
    assert passage_embeddings_fixture.title() == "Sample Passage Title"
    assert passage_embeddings_fixture.model() == DEFAULT_COLBERT_MODEL
    assert passage_embeddings_fixture.dim() == DEFAULT_COLBERT_DIM

def test_add_and_get_token_embeddings(passage_embeddings_fixture, token_embeddings):
    passage_embeddings_fixture.add_token_embeddings(token_embeddings)
    assert passage_embeddings_fixture.get_all_token_embeddings()[0] == token_embeddings
