import logging

import pytest
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# Uncomment to enable debug logging on Astra calls
# logging.getLogger('astrapy.utils').setLevel(logging.DEBUG)

def get_required_env(name) -> str:
    if name not in os.environ:
        pytest.skip(f"Missing required environment variable: {name}")
    return os.environ[name]
