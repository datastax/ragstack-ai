from __future__ import annotations

import math
import secrets
from typing import Callable, Iterable, Iterator

import numpy as np
import pytest
from dotenv import load_dotenv
from ragstack_knowledge_store import EmbeddingModel
from ragstack_knowledge_store.graph_store import GraphStore, MetadataIndexingType, Node
from ragstack_knowledge_store.links import Link
from ragstack_tests_utils import LocalCassandraTestStore

load_dotenv()

KEYSPACE = "default_keyspace"

vector_size = 52


def text_to_embedding(text: str) -> list[float]:
    """Embeds text using a simple ascii conversion algorithm"""
    embedding = np.zeros(vector_size)
    for i, char in enumerate(text):
        if i >= vector_size - 2:
            break
        embedding[i + 2] = ord(char) / 255  # Normalize ASCII value
    vector: list[float] = embedding.tolist()
    return vector


def angle_to_embedding(angle: float) -> list[float]:
    """Embeds angles onto a circle"""
    embedding = np.zeros(vector_size)
    embedding[0] = math.cos(angle * math.pi)
    embedding[1] = math.sin(angle * math.pi)
    vector: list[float] = embedding.tolist()
    return vector


class SimpleEmbeddingModel(EmbeddingModel):
    """
    Embeds numeric values (as strings in units of pi) into two-dimensional vectors on
    a circle, and other text into a simple 50-dimension vector.
    """

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Make a list of texts into a list of embedding vectors.
        """
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
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

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Make a list of texts into a list of embedding vectors.
        """
        return self.embed_texts(texts=texts)

    async def aembed_query(self, text: str) -> list[float]:
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

    def _make_graph_store(
        metadata_indexing: MetadataIndexingType = "all",
    ) -> GraphStore:
        name = secrets.token_hex(8)

        node_table = f"nodes_{name}"
        return GraphStore(
            embedding,
            session=session,
            keyspace=KEYSPACE,
            node_table=node_table,
            metadata_indexing=metadata_indexing,
        )

    yield _make_graph_store
    session.shutdown()


def _result_ids(nodes: Iterable[Node]) -> list[str]:
    return [n.id for n in nodes if n.id is not None]


def test_mmr_traversal(
    graph_store_factory: Callable[[MetadataIndexingType], GraphStore],
) -> None:
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

    v0 = Node(
        id="v0",
        text="-0.124",
        links={Link(direction="out", kind="explicit", tag="link")},
        metadata={"even": True},
    )
    v1 = Node(
        id="v1",
        text="+0.127",
        metadata={"even": False},
    )
    v2 = Node(
        id="v2",
        text="+0.25",
        links={Link(direction="in", kind="explicit", tag="link")},
        metadata={"even": True},
    )
    v3 = Node(
        id="v3",
        text="+1.0",
        links={Link(direction="in", kind="explicit", tag="link")},
        metadata={"even": False},
    )

    gs = graph_store_factory("all")
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

    # with metadata_filter even=True we should only get the `even` documents.
    results = gs.mmr_traversal_search("0.0", k=4, metadata_filter={"even": True})
    assert _result_ids(results) == ["v0", "v2"]

    # with initial_roots=[v0], we should start traversal there. this means that
    # the initial candidates are `v2`,`v3`. `v1` is unreachable and not
    # included.
    results = gs.mmr_traversal_search("0.0", fetch_k=0, k=4, initial_roots=["v0"])
    assert _result_ids(results) == ["v2", "v3"]

    # with initial_roots=[v1], we should start traversal there.
    # there are no adjacent nodes, so there are no results.
    results = gs.mmr_traversal_search("0.0", fetch_k=0, k=4, initial_roots=["v1"])
    assert _result_ids(results) == []

    # with initial_roots=[v0] and `fetch_k > 0` we should be able to reach everything.
    # but we don't re-fetch `v0`.
    results = gs.mmr_traversal_search("0.0", fetch_k=2, k=4, initial_roots=["v0"])
    assert _result_ids(results) == ["v1", "v3", "v2"]

    results = gs.mmr_traversal_search(
        "0.0", k=2, fetch_k=2, tag_filter={("explicit", "link")}
    )
    assert _result_ids(results) == ["v0", "v2"]

    results = gs.mmr_traversal_search(
        "0.0", k=2, fetch_k=2, tag_filter={("no", "match")}
    )
    assert _result_ids(results) == []


def test_write_retrieve_keywords(
    graph_store_factory: Callable[[MetadataIndexingType], GraphStore],
) -> None:
    greetings = Node(
        id="greetings",
        text="Typical Greetings",
        links={
            Link(direction="in", kind="parent", tag="parent"),
        },
        metadata={"Hello": False, "Greeting": "typical"},
    )
    doc1 = Node(
        id="doc1",
        text="Hello World",
        links={
            Link(direction="out", kind="parent", tag="parent"),
            Link(direction="bidir", kind="kw", tag="greeting"),
            Link(direction="bidir", kind="kw", tag="world"),
        },
        metadata={"Hello": True, "Greeting": "world"},
    )
    doc2 = Node(
        id="doc2",
        text="Hello Earth",
        links={
            Link(direction="out", kind="parent", tag="parent"),
            Link(direction="bidir", kind="kw", tag="greeting"),
            Link(direction="bidir", kind="kw", tag="earth"),
        },
        metadata={"Hello": True, "Greeting": "earth"},
    )

    gs = graph_store_factory("all")
    gs.add_nodes([greetings, doc1, doc2])

    # Doc2 is more similar, but World and Earth are similar enough that doc1 also shows
    # up.
    results = gs.similarity_search(text_to_embedding("Earth"), k=2)
    assert _result_ids(results) == ["doc2", "doc1"]

    results = gs.similarity_search(text_to_embedding("Earth"), k=1)
    assert _result_ids(results) == ["doc2"]

    # with metadata filter
    results = gs.similarity_search(
        text_to_embedding("Earth"), k=1, metadata_filter={"Greeting": "world"}
    )
    assert _result_ids(results) == ["doc1"]

    results = gs.traversal_search("Earth", k=2, depth=0)
    assert _result_ids(results) == ["doc2", "doc1"]

    results = gs.traversal_search("Earth", k=2, depth=1)
    assert _result_ids(results) == ["doc2", "doc1", "greetings"]

    # with metadata filter
    results = gs.traversal_search(
        "Earth", k=2, depth=1, metadata_filter={"Hello": True}
    )
    assert _result_ids(results) == ["doc2", "doc1"]

    # K=1 only pulls in doc2 (Hello Earth)
    results = gs.traversal_search("Earth", k=1, depth=0)
    assert _result_ids(results) == ["doc2"]

    # K=1 only pulls in doc2 (Hello Earth). Depth=1 traverses to parent and via keyword
    # edge.
    results = gs.traversal_search("Earth", k=1, depth=1)
    assert set(_result_ids(results)) == {"doc2", "doc1", "greetings"}


def test_metadata(
    graph_store_factory: Callable[[MetadataIndexingType], GraphStore],
) -> None:
    gs = graph_store_factory("all")
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


def test_graph_store_metadata(
    graph_store_factory: Callable[[MetadataIndexingType], GraphStore],
) -> None:
    gs = graph_store_factory("all")

    gs.add_nodes([Node(text="bb1", id="row1")])
    gotten1 = gs.get_node(content_id="row1")
    assert gotten1 == Node(text="bb1", id="row1", metadata={})

    gs.add_nodes([Node(text="bb2", id="row2", metadata={})])
    gotten2 = gs.get_node(content_id="row2")
    assert gotten2 == Node(text="bb2", id="row2", metadata={})

    md3 = {"a": 1, "b": "Bee", "c": True}
    gs.add_nodes([Node(text="bb3", id="row3", metadata=md3)])
    gotten3 = gs.get_node(content_id="row3")
    assert gotten3 == Node(text="bb3", id="row3", metadata=md3)

    md4 = {"c1": True, "c2": True, "c3": True}
    gs.add_nodes([Node(text="bb4", id="row4", metadata=md4)])
    gotten4 = gs.get_node(content_id="row4")
    assert gotten4 == Node(text="bb4", id="row4", metadata=md4)

    # metadata searches:
    md_gotten3a = list(gs.metadata_search(metadata={"a": 1}))[0]  # noqa: RUF015
    assert md_gotten3a == gotten3
    md_gotten3b = list(gs.metadata_search(metadata={"b": "Bee", "c": True}))[0]  # noqa: RUF015
    assert md_gotten3b == gotten3
    md_gotten4 = list(gs.metadata_search(metadata={"c1": True, "c3": True}))[0]  # noqa: RUF015
    assert md_gotten4 == gotten4

    # 'search' proper
    gs.add_nodes(
        [
            Node(text="ta", id="twin_a", metadata={"twin": True, "index": 0}),
            Node(text="tb", id="twin_b", metadata={"twin": True, "index": 1}),
        ]
    )
    md_twins_gotten = sorted(
        gs.metadata_search(metadata={"twin": True}),
        key=lambda res: int(float(res.metadata["index"])),
    )
    expected = [
        Node(text="ta", id="twin_a", metadata={"twin": True, "index": 0}),
        Node(text="tb", id="twin_b", metadata={"twin": True, "index": 1}),
    ]
    assert md_twins_gotten == expected
    assert list(gs.metadata_search(metadata={"fake": True})) == []


def test_graph_store_metadata_routing(
    graph_store_factory: Callable[[MetadataIndexingType], GraphStore],
) -> None:
    test_md = {"mds": "string", "mdn": 255, "mdb": True}

    gs_all = graph_store_factory("all")
    gs_all.add_nodes([Node(id="row1", text="bb1", metadata=test_md)])
    gotten_all = list(gs_all.metadata_search(metadata={"mds": "string"}))[0]  # noqa: RUF015
    assert gotten_all.metadata == test_md
    gs_none = graph_store_factory("none")
    gs_none.add_nodes([Node(id="row1", text="bb1", metadata=test_md)])
    with pytest.raises(ValueError):  # noqa: PT011
        # querying on non-indexed metadata fields:
        list(gs_none.metadata_search(metadata={"mds": "string"}))
    gotten_none = gs_none.get_node(content_id="row1")
    assert gotten_none is not None
    assert gotten_none.metadata == test_md
    test_md_allowdeny = {
        "mdas": "MDAS",
        "mdds": "MDDS",
        "mdan": 255,
        "mddn": 127,
        "mdab": True,
        "mddb": True,
    }
    gs_allow = graph_store_factory(("allow", {"mdas", "mdan", "mdab"}))
    gs_allow.add_nodes([Node(id="row1", text="bb1", metadata=test_md_allowdeny)])
    with pytest.raises(ValueError):  # noqa: PT011
        list(gs_allow.metadata_search(metadata={"mdds": "MDDS"}))
    gotten_allow = list(gs_allow.metadata_search(metadata={"mdas": "MDAS"}))[0]  # noqa: RUF015
    assert gotten_allow.metadata == test_md_allowdeny
    gs_deny = graph_store_factory(("deny", {"mdds", "mddn", "mddb"}))
    gs_deny.add_nodes([Node(id="row1", text="bb1", metadata=test_md_allowdeny)])
    with pytest.raises(ValueError):  # noqa: PT011
        list(gs_deny.metadata_search(metadata={"mdds": "MDDS"}))
    gotten_deny = list(gs_deny.metadata_search(metadata={"mdas": "MDAS"}))[0]  # noqa: RUF015
    assert gotten_deny.metadata == test_md_allowdeny
