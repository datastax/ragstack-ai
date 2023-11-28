import logging
import uuid

import pytest
import os


def random_string():
    return str(uuid.uuid4()).split("-")[0]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


# Uncomment to enable debug logging on Astra calls
# logging.getLogger('astrapy.utils').setLevel(logging.DEBUG)


def get_required_env(name) -> str:
    if name not in os.environ:
        pytest.skip(f"Missing required environment variable: {name}")
    return os.environ[name]


# astra dev
os.environ["ASTRA_DEV_TABLE_NAME"] = f"documents_{random_string()}"

# astra prod
os.environ["ASTRA_PROD_TABLE_NAME"] = f"documents_{random_string()}"

# azure-open-ai
os.environ["AZURE_OPEN_AI_CHAT_MODEL_DEPLOYMENT"] = "gpt-35-turbo"
os.environ["AZURE_OPEN_AI_EMBEDDINGS_MODEL_DEPLOYMENT"] = "text-embedding-ada-002"

# vertex-ai
with open("/tmp/gcloud-account-key.json", "w") as f:
    f.write(os.getenv("GCLOUD_ACCOUNT_KEY_JSON", ""))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcloud-account-key.json"
