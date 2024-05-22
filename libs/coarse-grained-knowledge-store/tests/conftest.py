import secrets
from typing import Iterable, Iterator, Optional

import pytest
from cassandra.cluster import Cluster, Session
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs

from ragstack_coarse_grained_knowledge_store.knowledge_store import KnowledgeStore


@pytest.fixture(scope="session")
def db_keyspace() -> str:
    return "default_keyspace"


@pytest.fixture(scope="session")
def cassandra_port(db_keyspace: str) -> Iterator[int]:
    # TODO: Allow running against local Cassandra and/or Astra using pytest option.
    cassandra = DockerContainer("cassandra:5")
    cassandra.with_exposed_ports(9042)
    cassandra.with_env(
        "JVM_OPTS",
        "-Dcassandra.skip_wait_for_gossip_to_settle=0 -Dcassandra.initial_token=0",
    )
    cassandra.with_env("HEAP_NEWSIZE", "128M")
    cassandra.with_env("MAX_HEAP_SIZE", "1024M")
    cassandra.with_env("CASSANDRA_ENDPOINT_SNITCH", "GossipingPropertyFileSnitch")
    cassandra.with_env("CASSANDRA_DC", "datacenter1")
    cassandra.start()
    wait_for_logs(cassandra, "Startup complete")
    cassandra.get_wrapped_container().exec_run(
        (
            f"""cqlsh -e "CREATE KEYSPACE {db_keyspace} WITH replication = """
            '''{'class': 'SimpleStrategy', 'replication_factor': '1'};"'''
        )
    )
    port = cassandra.get_exposed_port(9042)
    print(f"Cassandra started. Port is {port}")
    yield port
    cassandra.stop()


@pytest.fixture(scope="session")
def db_session(cassandra_port: int) -> Session:
    print(f"Connecting to cassandra on {cassandra_port}")
    cluster = Cluster(
        port=cassandra_port,
    )
    return cluster.connect()


@pytest.fixture(scope="session")
def openai_embedding() -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings()


@pytest.fixture(scope="session")
def local_embedding() -> Embeddings:
    from langchain_community.embeddings.ollama import OllamaEmbeddings

    return OllamaEmbeddings()


class DataFixture:
    def __init__(self, session: Session, keyspace: str, embedding: Embeddings) -> None:
        self.session = session
        self.keyspace = "default_keyspace"
        self.uid = secrets.token_hex(8)
        self.node_table = f"nodes_{self.uid}"
        self.edge_table = f"edges_{self.uid}"
        self.embedding = embedding
        self._store = None

    def store(
        self, initial_documents: Iterable[Document] = [], ids: Optional[Iterable[str]] = None
    ) -> KnowledgeStore:
        if initial_documents and self._store is not None:
            raise ValueError("Store already initialized")
        elif self._store is None:
            self._store = KnowledgeStore.from_documents(
                initial_documents,
                self.embedding,
                session=self.session,
                keyspace=self.keyspace,
                node_table=self.node_table,
                edge_table=self.edge_table,
                ids=ids,
            )

        return self._store

    def drop(self):
        self.session.execute(f"DROP TABLE IF EXISTS {self.keyspace}.{self.node_table};")
        self.session.execute(f"DROP TABLE IF EXISTS {self.keyspace}.{self.edge_table};")


@pytest.fixture()
def fresh_fixture(
    db_session: Session, db_keyspace: str, openai_embedding: Embeddings
) -> Iterator[DataFixture]:
    data = DataFixture(session=db_session, keyspace=db_keyspace, embedding=openai_embedding)
    yield data
    data.drop()
