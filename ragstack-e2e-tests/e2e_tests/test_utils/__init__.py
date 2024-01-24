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


def random_string():
    return str(uuid.uuid4()).split("-")[0]
