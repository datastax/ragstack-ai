import math
import secrets
from typing import Callable, Iterable, Iterator, List

import numpy as np
import pytest
from dotenv import load_dotenv
from ragstack_knowledge_store import EmbeddingModel
from ragstack_knowledge_store.graph_store import GraphStore, Node
from ragstack_knowledge_store.links import Link
from ragstack_tests_utils import LocalCassandraTestStore

load_dotenv()

KEYSPACE = "default_keyspace"

vector_size = 52


def text_to_embedding(text: str) -> List[float]:
    """Embeds text using a simple ascii conversion algorithm"""
    embedding = np.zeros(vector_size)
    for i, char in enumerate(text):
        if i >= vector_size - 2:
            break
        embedding[i + 2] = ord(char) / 255  # Normalize ASCII value
    vector: List[float] = embedding.tolist()
    return vector


def angle_to_embedding(angle: float) -> List[float]:
    """Embeds angles onto a circle"""
    embedding = np.zeros(vector_size)
    embedding[0] = math.cos(angle * math.pi)
    embedding[1] = math.sin(angle * math.pi)
    vector: List[float] = embedding.tolist()
    return vector


class SimpleEmbeddingModel(EmbeddingModel):
    """
    Embeds numeric values (as strings in units of pi) into two-dimensional vectors on
    a circle, and other text into a simple 50-dimension vector.
    """

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Make a list of texts into a list of embedding vectors.
        """
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        Convert input text to a 'vector' (list of floats).
        If the text is a number, use it as the angle for the
        unit vector in units of pi.
        Any other input text is embedded as is.
        """
        try:
            angle = float(text)
            return angle_to_embedding(angle=angle)
        except ValueError:
            # Assume: just test string
            return text_to_embedding(text)

    async def aembed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Make a list of texts into a list of embedding vectors.
        """
        return self.embed_texts(texts=texts)

    async def aembed_query(self, text: str) -> List[float]:
        """
        Convert input text to a 'vector' (list of floats).
        If the text is a number, use it as the angle for the
        unit vector in units of pi.
        Any other input text is embedded as is.
        """
        return self.embed_query(text=text)


@pytest.fixture(scope="session")
def cassandra() -> Iterator[LocalCassandraTestStore]:
    store = LocalCassandraTestStore()
    yield store

    if store.docker_container:
        store.docker_container.stop()


@pytest.fixture()
def graph_store_factory(
    cassandra: LocalCassandraTestStore,
) -> Iterator[Callable[[], GraphStore]]:
    session = cassandra.create_cassandra_session()
    session.set_keyspace(KEYSPACE)

    embedding = SimpleEmbeddingModel()

    def _make_graph_store() -> GraphStore:
        name = secrets.token_hex(8)

        node_table = f"nodes_{name}"
        return GraphStore(
            embedding,
            session=session,
            keyspace=KEYSPACE,
            node_table=node_table,
        )

    yield _make_graph_store

    session.shutdown()


def _result_ids(nodes: Iterable[Node]) -> List[str]:
    return [n.id for n in nodes if n.id is not None]


def test_graph_store_creation(graph_store_factory: Callable[[], GraphStore]) -> None:
    """Test that a graph store can be created.

    This verifies the schema can be applied and the queries prepared.
    """
    graph_store_factory()


def test_mmr_traversal(graph_store_factory: Callable[[], GraphStore]) -> None:
    """
    Test end to end construction and MMR search.
    The embedding function used here ensures `texts` become
    the following vectors on a circle (numbered v0 through v3):

           ______ v2
          /      \
         /        |  v1
    v3  |     .    | query
         |        /  v0
          |______/                 (N.B. very crude drawing)

    With fetch_k==2 and k==2, when query is at (1, ),
    one expects that v2 and v0 are returned (in some order)
    because v1 is "too close" to v0 (and v0 is closer than v1)).

    Both v2 and v3 are reachable via edges from v0, so once it is
    selected, those are both considered.
    """
    gs = graph_store_factory()

    v0 = Node(
        id="v0",
        text="-0.124",
        links={Link(direction="out", kind="explicit", tag="link")},
    )
    v1 = Node(
        id="v1",
        text="+0.127",
    )
    v2 = Node(
        id="v2",
        text="+0.25",
        links={Link(direction="in", kind="explicit", tag="link")},
    )
    v3 = Node(
        id="v3",
        text="+1.0",
        links={Link(direction="in", kind="explicit", tag="link")},
    )
    gs.add_nodes([v0, v1, v2, v3])

    results = gs.mmr_traversal_search("0.0", k=2, fetch_k=2)
    assert _result_ids(results) == ["v0", "v2"]

    # With max depth 0, no edges are traversed, so this doesn't reach v2 or v3.
    # So it ends up picking "v1" even though it's similar to "v0".
    results = gs.mmr_traversal_search("0.0", k=2, fetch_k=2, depth=0)
    assert _result_ids(results) == ["v0", "v1"]

    # With max depth 0 but higher `fetch_k`, we encounter v2
    results = gs.mmr_traversal_search("0.0", k=2, fetch_k=3, depth=0)
    assert _result_ids(results) == ["v0", "v2"]

    # v0 score is .46, v2 score is 0.16 so it won't be chosen.
    results = gs.mmr_traversal_search("0.0", k=2, score_threshold=0.2)
    assert _result_ids(results) == ["v0"]

    # with k=4 we should get all of the documents.
    results = gs.mmr_traversal_search("0.0", k=4)
    assert _result_ids(results) == ["v0", "v2", "v1", "v3"]


def test_write_retrieve_keywords(graph_store_factory: Callable[[], GraphStore]) -> None:
    gs = graph_store_factory()

    greetings = Node(
        id="greetings",
        text="Typical Greetings",
        links={
            Link(direction="in", kind="parent", tag="parent"),
        },
    )
    doc1 = Node(
        id="doc1",
        text="Hello World",
        links={
            Link(direction="out", kind="parent", tag="parent"),
            Link(direction="bidir", kind="kw", tag="greeting"),
            Link(direction="bidir", kind="kw", tag="world"),
        },
    )
    doc2 = Node(
        id="doc2",
        text="Hello Earth",
        links={
            Link(direction="out", kind="parent", tag="parent"),
            Link(direction="bidir", kind="kw", tag="greeting"),
            Link(direction="bidir", kind="kw", tag="earth"),
        },
    )

    gs.add_nodes([greetings, doc1, doc2])

    # Doc2 is more similar, but World and Earth are similar enough that doc1 also shows
    # up.
    results = gs.similarity_search(text_to_embedding("Earth"), k=2)
    assert _result_ids(results) == ["doc2", "doc1"]

    results = gs.similarity_search(text_to_embedding("Earth"), k=1)
    assert _result_ids(results) == ["doc2"]

    results = gs.traversal_search("Earth", k=2, depth=0)
    assert _result_ids(results) == ["doc2", "doc1"]

    results = gs.traversal_search("Earth", k=2, depth=1)
    assert _result_ids(results) == ["doc2", "doc1", "greetings"]

    # K=1 only pulls in doc2 (Hello Earth)
    results = gs.traversal_search("Earth", k=1, depth=0)
    assert _result_ids(results) == ["doc2"]

    # K=1 only pulls in doc2 (Hello Earth). Depth=1 traverses to parent and via keyword
    # edge.
    results = gs.traversal_search("Earth", k=1, depth=1)
    assert set(_result_ids(results)) == {"doc2", "doc1", "greetings"}


def test_metadata(graph_store_factory: Callable[[], GraphStore]) -> None:
    gs = graph_store_factory()

    gs.add_nodes(
        [
            Node(
                id="a",
                text="A",
                links={
                    Link(direction="in", kind="hyperlink", tag="http://a"),
                    Link(direction="bidir", kind="other", tag="foo"),
                },
                metadata={"other": "some other field"},
            )
        ]
    )
    results = list(gs.similarity_search(text_to_embedding("A")))
    assert len(results) == 1
    assert results[0].id == "a"
    assert results[0].metadata["other"] == "some other field"
    assert results[0].links == {
        Link(direction="in", kind="hyperlink", tag="http://a"),
        Link(direction="bidir", kind="other", tag="foo"),
    }
