import math
import secrets
from typing import Iterable, List, Optional

import pytest
from cassandra.cluster import Session
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from ragstack_langchain.graph_store import CassandraGraphStore
from ragstack_langchain.graph_store.base import METADATA_CONTENT_ID_KEY
from ragstack_langchain.graph_store.links import METADATA_LINKS_KEY, Link
from ragstack_tests_utils.test_store import KEYSPACE

from .conftest import get_astradb_test_store, get_local_cassandra_test_store


class GraphStoreFactory:
    def __init__(self, session: Session, keyspace: str, embedding: Embeddings) -> None:
        self.session = session
        self.keyspace = keyspace
        self.uid = secrets.token_hex(8)
        self.node_table = f"nodes_{self.uid}"
        self.targets_table = f"targets_{self.uid}"
        self.embedding = embedding
        self._store = None

    def store(
        self,
        initial_documents: Iterable[Document] = (),
        ids: Optional[Iterable[str]] = None,
        embedding: Optional[Embeddings] = None,
    ) -> CassandraGraphStore:
        if initial_documents and self._store is not None:
            raise ValueError("Store already initialized")
        elif self._store is None:
            self._store = CassandraGraphStore.from_documents(
                initial_documents,
                embedding=embedding or self.embedding,
                session=self.session,
                keyspace=self.keyspace,
                node_table=self.node_table,
                targets_table=self.targets_table,
                ids=ids,
            )

        return self._store

    def drop(self):
        self.session.execute(f"DROP TABLE IF EXISTS {self.keyspace}.{self.node_table};")
        self.session.execute(
            f"DROP TABLE IF EXISTS {self.keyspace}.{self.targets_table};"
        )


@pytest.fixture(scope="session")
def openai_embedding() -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings()


@pytest.fixture()
def cassandra(openai_embedding: Embeddings):
    vstore = get_local_cassandra_test_store()
    session = vstore.create_cassandra_session()
    gs_factory = GraphStoreFactory(
        session=session, keyspace=KEYSPACE, embedding=openai_embedding
    )
    yield gs_factory
    gs_factory.drop()


@pytest.fixture()
def astra_db(openai_embedding: Embeddings):
    vstore = get_astradb_test_store()
    session = vstore.create_cassandra_session()
    gs_factory = GraphStoreFactory(
        session=session, keyspace=KEYSPACE, embedding=openai_embedding
    )
    yield gs_factory
    gs_factory.drop()


class AngularTwoDimensionalEmbeddings(Embeddings):
    """
    From angles (as strings in units of pi) to unit embedding vectors on a circle.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Make a list of texts into a list of embedding vectors.
        """
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        Convert input text to a 'vector' (list of floats).
        If the text is a number, use it as the angle for the
        unit vector in units of pi.
        Any other input text becomes the singular result [0, 0] !
        """
        try:
            angle = float(text)
            return [math.cos(angle * math.pi), math.sin(angle * math.pi)]
        except ValueError:
            # Assume: just test string, no attention is paid to values.
            return [0.0, 0.0]


def _result_ids(docs: Iterable[Document]) -> List[str]:
    return [d.metadata[METADATA_CONTENT_ID_KEY] for d in docs]


@pytest.mark.parametrize("gs_factory", ["cassandra", "astra_db"])
def test_mmr_traversal(request, gs_factory: str):
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
    gs_factory = request.getfixturevalue(gs_factory)
    store = gs_factory.store(
        embedding=AngularTwoDimensionalEmbeddings(),
    )

    v0 = Document(
        page_content="-0.124",
        metadata={
            METADATA_CONTENT_ID_KEY: "v0",
            METADATA_LINKS_KEY: {
                Link.outgoing(kind="explicit", tag="link"),
            },
        },
    )
    v1 = Document(
        page_content="+0.127",
        metadata={
            METADATA_CONTENT_ID_KEY: "v1",
        },
    )
    v2 = Document(
        page_content="+0.25",
        metadata={
            METADATA_CONTENT_ID_KEY: "v2",
            METADATA_LINKS_KEY: {
                Link.incoming(kind="explicit", tag="link"),
            },
        },
    )
    v3 = Document(
        page_content="+1.0",
        metadata={
            METADATA_CONTENT_ID_KEY: "v3",
            METADATA_LINKS_KEY: {
                Link.incoming(kind="explicit", tag="link"),
            },
        },
    )
    store.add_documents([v0, v1, v2, v3])

    results = store.mmr_traversal_search("0.0", k=2, fetch_k=2)
    assert _result_ids(results) == ["v0", "v2"]

    # With max depth 0, no edges are traversed, so this doesn't reach v2 or v3.
    # So it ends up picking "v1" even though it's similar to "v0".
    results = store.mmr_traversal_search("0.0", k=2, fetch_k=2, depth=0)
    assert _result_ids(results) == ["v0", "v1"]

    # With max depth 0 but higher `fetch_k`, we encounter v2
    results = store.mmr_traversal_search("0.0", k=2, fetch_k=3, depth=0)
    assert _result_ids(results) == ["v0", "v2"]

    # v0 score is .46, v2 score is 0.16 so it won't be chosen.
    results = store.mmr_traversal_search("0.0", k=2, score_threshold=0.2)
    assert _result_ids(results) == ["v0"]

    # with k=4 we should get all of the documents.
    results = store.mmr_traversal_search("0.0", k=4)
    assert _result_ids(results) == ["v0", "v2", "v1", "v3"]


@pytest.mark.parametrize("gs_factory", ["cassandra", "astra_db"])
def test_write_retrieve_keywords(request, gs_factory: str):
    gs_factory = request.getfixturevalue(gs_factory)
    greetings = Document(
        page_content="Typical Greetings",
        metadata={
            METADATA_CONTENT_ID_KEY: "greetings",
            METADATA_LINKS_KEY: {
                Link.incoming(kind="parent", tag="parent"),
            },
        },
    )
    doc1 = Document(
        page_content="Hello World",
        metadata={
            METADATA_CONTENT_ID_KEY: "doc1",
            METADATA_LINKS_KEY: {
                Link.outgoing(kind="parent", tag="parent"),
                Link.bidir(kind="kw", tag="greeting"),
                Link.bidir(kind="kw", tag="world"),
            },
        },
    )
    doc2 = Document(
        page_content="Hello Earth",
        metadata={
            METADATA_CONTENT_ID_KEY: "doc2",
            METADATA_LINKS_KEY: {
                Link.outgoing(kind="parent", tag="parent"),
                Link.bidir(kind="kw", tag="greeting"),
                Link.bidir(kind="kw", tag="earth"),
            },
        },
    )

    store = gs_factory.store([greetings, doc1, doc2])

    # Doc2 is more similar, but World and Earth are similar enough that doc1 also shows
    # up.
    results = store.similarity_search("Earth", k=2)
    assert _result_ids(results) == ["doc2", "doc1"]

    results = store.similarity_search("Earth", k=1)
    assert _result_ids(results) == ["doc2"]

    results = store.traversal_search("Earth", k=2, depth=0)
    assert _result_ids(results) == ["doc2", "doc1"]

    results = store.traversal_search("Earth", k=2, depth=1)
    assert _result_ids(results) == ["doc2", "doc1", "greetings"]

    # K=1 only pulls in doc2 (Hello Earth)
    results = store.traversal_search("Earth", k=1, depth=0)
    assert _result_ids(results) == ["doc2"]

    # K=1 only pulls in doc2 (Hello Earth). Depth=1 traverses to parent and via keyword
    # edge.
    results = store.traversal_search("Earth", k=1, depth=1)
    assert set(_result_ids(results)) == {"doc2", "doc1", "greetings"}


@pytest.mark.parametrize("gs_factory", ["cassandra", "astra_db"])
def test_metadata(request, gs_factory: str):
    gs_factory: GraphStoreFactory = request.getfixturevalue(gs_factory)
    store = gs_factory.store(
        [
            Document(
                page_content="A",
                metadata={
                    METADATA_CONTENT_ID_KEY: "a",
                    METADATA_LINKS_KEY: {
                        Link.incoming(kind="hyperlink", tag="http://a"),
                        Link.bidir(kind="other", tag="foo"),
                    },
                    "other": "some other field",
                },
            )
        ]
    )
    results = store.similarity_search("A")
    assert len(results) == 1
    metadata = results[0].metadata
    assert metadata["other"] == "some other field"
    assert metadata[METADATA_CONTENT_ID_KEY] == "a"
    assert set(metadata[METADATA_LINKS_KEY]) == {
        Link.incoming(kind="hyperlink", tag="http://a"),
        Link.bidir(kind="other", tag="foo"),
    }
