import secrets
from typing import Callable, Iterator, List

import pytest
from dotenv import load_dotenv
from ragstack_knowledge_store import EmbeddingModel
from ragstack_knowledge_store.graph_store import GraphStore
from ragstack_tests_utils import LocalCassandraTestStore

load_dotenv()

KEYSPACE = "default_keyspace"


@pytest.fixture(scope="session")
def cassandra() -> Iterator[LocalCassandraTestStore]:
    store = LocalCassandraTestStore()
    yield store

    if store.docker_container:
        store.docker_container.stop()


class DummyEmbeddingModel(EmbeddingModel):
    def embed_texts(self, _: List[str]) -> List[List[float]]:
        return []

    def embed_query(self, _: str) -> List[float]:
        return []

    async def aembed_texts(self, _: List[str]) -> List[List[float]]:
        return []

    async def aembed_query(self, _: str) -> List[float]:
        return []


@pytest.fixture()
def graph_store_factory(
    cassandra: LocalCassandraTestStore,
) -> Iterator[Callable[[], GraphStore]]:
    session = cassandra.create_cassandra_session()
    session.set_keyspace(KEYSPACE)

    embedding = DummyEmbeddingModel()

    def _make_graph_store() -> GraphStore:
        name = secrets.token_hex(8)

        node_table = f"nodes_{name}"
        targets_table = f"targets_{name}"
        return GraphStore(
            embedding,
            session=session,
            keyspace=KEYSPACE,
            node_table=node_table,
            targets_table=targets_table,
        )

    yield _make_graph_store

    session.shutdown()


def test_graph_store_creation(graph_store_factory: Callable[[], GraphStore]) -> None:
    """Test that a graph store can be created.

    This verifies the schema can be applied and the queries prepared.
    """
    graph_store_factory()
