import logging
import os
import uuid

import pytest

LOGGER = logging.getLogger(__name__)


def get_required_env(name) -> str:
    if name not in os.environ:
        LOGGER.warning(f"Missing required environment variable: {name}")
        pytest.skip(f"Missing required environment variable: {name}")
    value = os.environ[name]
    if not value:
        LOGGER.warning(f"Empty required environment variable: {name}")
        pytest.skip(f"Empty required environment variable: {name}")
    return value


def get_local_resource_path(filename: str):
    dirname = os.path.dirname(__file__)
    e2e_tests_dir = os.path.dirname(dirname)
    return os.path.join(e2e_tests_dir, "resources", filename)


def random_string() -> str:
    return str(uuid.uuid4()).split("-")[0]


def skip_test_due_to_implementation_not_supported(implementation: str) -> None:
    pytest.skip(f"Skipping test because {implementation} is not supported")


def is_skipped_due_to_implementation_not_supported(error: str) -> bool:
    return "Skipping test because" in error and "is not supported" in error
