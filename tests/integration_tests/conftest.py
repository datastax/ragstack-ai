import logging
import os
from abc import abstractmethod, ABC

import cassio
import pytest
from cassandra.cluster import Cluster, PlainTextAuthProvider, Session

from tests.integration_tests.cassandra_container import CassandraContainer

KEYSPACE = "colbert"


class TestStore(ABC):
    @abstractmethod
    def create_cassandra_session(self) -> Session:
        pass


class LocalCassandraTestStore(TestStore):
    def __init__(self):
        super().__init__()
        start_container = os.environ.get("CASSANDRA_START_CONTAINER", "true")
        self.port = 9042
        self.docker_container = None
        if start_container == "true":
            self.docker_container = CassandraContainer()
            logging.info("Starting Cassandra container")
            self.docker_container.start()
            self.port = self.docker_container.get_mapped_port()

    def create_cassandra_session(self) -> Session:
        cluster = Cluster(
            [("127.0.0.1", self.port)],
            auth_provider=PlainTextAuthProvider("cassandra", "cassandra"),
        )

        cassandra_session = cluster.connect()
        cassandra_session.execute(f"DROP KEYSPACE IF EXISTS {KEYSPACE}")
        cassandra_session.execute(
            f"CREATE KEYSPACE IF NOT EXISTS {KEYSPACE} WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': '1'}}"
        )
        cassio.init(session=cassandra_session)
        return cassandra_session


class AstraDBTestStore(TestStore):
    token: str
    database_id: str

    def __init__(self):
        super().__init__()
        if not os.getenv("ASTRA_DB_ID") or not os.getenv("ASTRA_DB_TOKEN"):
            raise ValueError(
                "ASTRA_DB_ID and ASTRA_DB_TOKEN environment variables must be set"
            )

    def create_cassandra_session(self) -> Session:
        id = os.getenv("ASTRA_DB_ID")
        token = os.getenv("ASTRA_DB_TOKEN")
        cassio.init(token=token, database_id=id, keyspace=KEYSPACE)
        session = cassio.config.resolve_session()
        session.execute(f"DROP KEYSPACE IF EXISTS {KEYSPACE}")
        session.execute(
            f"CREATE KEYSPACE IF NOT EXISTS {KEYSPACE} WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': '1'}}"
        )
        return session


status = {
    "local_cassandra_test_store": None,
    "astradb_test_store": None,
}


def get_local_cassandra_test_store():
    if not status["local_cassandra_test_store"]:
        status["local_cassandra_test_store"] = LocalCassandraTestStore()
    return status["local_cassandra_test_store"]


def get_astradb_test_store():
    if not status["astradb_test_store"]:
        status["astradb_test_store"] = AstraDBTestStore()
    return status["astradb_test_store"]


@pytest.fixture(scope="session", autouse=True)
def before_after_tests():
    yield
    if (
        status["local_cassandra_test_store"]
        and status["local_cassandra_test_store"].docker_container
    ):
        status["local_cassandra_test_store"].docker_container.stop()
