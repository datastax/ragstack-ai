import pytest
from cassandra.cluster import Session
from ragstack_tests_utils import AstraDBTestStore, LocalCassandraTestStore


@pytest.fixture(scope="session")
def cassandra() -> LocalCassandraTestStore:
    store = LocalCassandraTestStore()
    yield store
    if store.docker_container:
        store.docker_container.stop()


@pytest.fixture(scope="session")
def astra_db() -> AstraDBTestStore:
    return AstraDBTestStore()


@pytest.fixture()
def session(request) -> Session:
    test_store = request.getfixturevalue(request.param)
    session = test_store.create_cassandra_session()
    session.default_timeout = 180
    return session
