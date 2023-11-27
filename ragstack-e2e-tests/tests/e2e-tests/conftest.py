import pytest
import os


def get_required_env(name) -> str:
    if name not in os.environ:
        pytest.skip(f"Missing required environment variable: {name}")
    return os.environ[name]
