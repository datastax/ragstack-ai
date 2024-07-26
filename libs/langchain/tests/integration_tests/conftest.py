from typing import Iterator

import pytest
from _pytest.fixtures import FixtureRequest
from cassandra.cluster import Session
from dotenv import load_dotenv
from ragstack_tests_utils import AstraDBTestStore, LocalCassandraTestStore

load_dotenv()


@pytest.fixture(scope="session")
def cassandra() -> Iterator[LocalCassandraTestStore]:
    store = LocalCassandraTestStore()
    yield store
    if store.docker_container:
        store.docker_container.stop()


@pytest.fixture(scope="session")
def astra_db() -> AstraDBTestStore:
    return AstraDBTestStore()


def get_session(request: FixtureRequest) -> Session:
    test_store = request.getfixturevalue(request.param)
    session = test_store.create_cassandra_session()
    session.default_timeout = 180
    return session


@pytest.fixture()
def session(request: FixtureRequest) -> Session:
    return get_session(request)
