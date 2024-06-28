import secrets
from typing import Iterator
from langchain_openai import OpenAIEmbeddings
import pytest
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs
from cassandra.cluster import Cluster
from dotenv import load_dotenv

from ragstack_knowledge_store.graph_store import GraphStore

load_dotenv()

KEYSPACE = "default_keyspace"

@pytest.fixture(scope="session")
def cassandra_port() -> Iterator[int]:
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
            f"""cqlsh -e "CREATE KEYSPACE {KEYSPACE} WITH replication = """
            '''{'class': 'SimpleStrategy', 'replication_factor': '1'};"'''
        )
    )
    port = cassandra.get_exposed_port(9042)
    print(f"Cassandra started. Port is {port}")
    yield port
    cassandra.stop()

@pytest.fixture
def graph_store_factory(cassandra_port: int):
    cluster = Cluster(
        port=cassandra_port,
    )
    session = cluster.connect()
    session.set_keyspace(KEYSPACE)

    embedding = OpenAIEmbeddings()

    def _make_graph_store():
        name = secrets.token_hex(8)

        node_table = f"nodes_{name}"
        targets_table = f"targets_{name}"
        return GraphStore(
            embedding,
            session = session,
            keyspace = KEYSPACE,
            node_table = node_table,
            targets_table = targets_table,
        )

    yield _make_graph_store

    session.shutdown()

def test_graph_store_creation(graph_store_factory):
    """Test that a graph store can be created.

    This verifies the schema can be applied and the queries prepared.
    """
    graph_store_factory()