import pytest

from e2e_tests.conftest import (
    get_vector_store_handler,
)
from e2e_tests.test_utils.vector_store_handler import (
    VectorStoreImplementation,
)


@pytest.fixture
def astra_db():
    handler = get_vector_store_handler(VectorStoreImplementation.ASTRADB)
    context = handler.before_test()
    yield context
    handler.after_test()


@pytest.fixture
def cassandra():
    handler = get_vector_store_handler(VectorStoreImplementation.CASSANDRA)
    context = handler.before_test()
    yield context
    handler.after_test()
