import logging
import os
import tempfile
import time

from astrapy.db import AstraDB


def get_required_env(name) -> str:
    if name not in os.environ:
        raise ValueError(f"Missing required environment variable: {name}")
    value = os.environ[name]
    if not value:
        raise ValueError(f"Empty required environment variable: {name}")
    return value


# vertex-ai
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    with tempfile.NamedTemporaryFile(
        prefix="gcloud-account-key", suffix=".json", mode="w", delete=False
    ) as f:
        f.write(os.getenv("GCLOUD_ACCOUNT_KEY_JSON", ""))
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name

client = AstraDB(
    token=get_required_env("ASTRA_DB_APPLICATION_TOKEN"),
    api_endpoint=get_required_env("ASTRA_DB_API_ENDPOINT"),
)


def try_delete_with_backoff(collection: str, sleep=1, max_tries=2) -> None:
    try:
        logging.info("deleting collection %s", collection)
        response = client.delete_collection(collection)
        logging.info("delete collection %s response: %s", collection, response)
    except Exception:
        max_tries -= 1
        if max_tries < 0:
            raise

        logging.warning(
            "An exception occurred deleting collection %s: ", collection, exc_info=True
        )
        time.sleep(sleep)
        try_delete_with_backoff(collection, sleep * 2, max_tries)


def before_notebook() -> None:
    collections = client.get_collections().get("status").get("collections")
    logging.info("Existing collections: %s", collections)
    for collection in collections:
        try_delete_with_backoff(collection)
