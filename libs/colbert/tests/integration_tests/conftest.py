import pytest
from ragstack_tests_utils import AstraDBTestStore, LocalCassandraTestStore

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


@pytest.hookimpl()
def pytest_sessionfinish():
    if (
        status["local_cassandra_test_store"]
        and status["local_cassandra_test_store"].docker_container
    ):
        status["local_cassandra_test_store"].docker_container.stop()
