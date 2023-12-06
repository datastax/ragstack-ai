import logging
import os
import uuid
from dataclasses import dataclass

import pytest
from astrapy.db import AstraDB as LibAstraDB


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


@dataclass
class AstraRef:
    token: str
    api_endpoint: str
    collection: str
    id: str


def get_astra_dev_ref() -> AstraRef:
    return AstraRef(
        token=get_required_env("ASTRA_DEV_DB_TOKEN"),
        api_endpoint=get_required_env("ASTRA_DEV_DB_ENDPOINT"),
        collection=get_required_env("ASTRA_DEV_TABLE_NAME"),
        id=get_required_env("ASTRA_DEV_DB_ID"),
    )


def get_astra_prod_ref() -> AstraRef:
    return AstraRef(
        token=get_required_env("ASTRA_PROD_DB_TOKEN"),
        api_endpoint=get_required_env("ASTRA_PROD_DB_ENDPOINT"),
        collection=get_required_env("ASTRA_PROD_TABLE_NAME"),
        id=get_required_env("ASTRA_PROD_DB_ID"),
    )


def get_default_astra_ref() -> AstraRef:
    return get_astra_prod_ref()


def delete_all_astra_collections(astra_ref: AstraRef):
    """
    Deletes all collections.

    Current AstraDB has a limit of 5 collections, meaning orphaned collections
    will cause subsequent tests to fail if the limit is reached.
    """
    raw_client = LibAstraDB(api_endpoint=astra_ref.api_endpoint, token=astra_ref.token)
    collections = raw_client.get_collections().get("status").get("collections")
    logging.info(f"Existing collections: {collections}")
    for collection_info in collections:
        logging.info(f"Deleting collection: {collection_info}")
        raw_client.delete_collection(collection_info)


def delete_astra_collection(astra_ref: AstraRef) -> None:
    raw_client = LibAstraDB(api_endpoint=astra_ref.api_endpoint, token=astra_ref.token)
    raw_client.delete_collection(astra_ref.collection)


failed_report_lines = []
all_report_lines = []
tests_stats = {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
}


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call":
        info = os.getenv("RAGSTACK_E2E_TESTS_TEST_INFO", "")
        if not info:
            info = os.path.basename(item.path) + "::" + item.name
        if rep.outcome == "failed":
            test_outcome = "❌"
            tests_stats["failed"] += 1
        elif rep.outcome == "passed":
            test_outcome = "✅"
            tests_stats["passed"] += 1
        elif rep.outcome == "skipped":
            test_outcome = "⚠️"
            tests_stats["skipped"] += 1
        else:
            test_outcome = f"(? {rep.outcome}))"
        result = " " + str(call.excinfo) if call.excinfo else ""
        report_line = f"{info} -> {test_outcome}{result}"
        if rep.outcome != "passed":
            # also keep skipped tests in the report
            failed_report_lines.append(report_line)
        all_report_lines.append(report_line)


def set_current_test_info_simple_rag(llm: str, embedding: str, vector_db: str) -> None:
    os.environ[
        "RAGSTACK_E2E_TESTS_TEST_INFO"
    ] = f"simple_rag::{llm},{embedding},{vector_db}"


def set_current_test_info_document_loader(doc_loader: str) -> None:
    os.environ["RAGSTACK_E2E_TESTS_TEST_INFO"] = f"document_loader::{doc_loader}"


@pytest.fixture(scope="session", autouse=True)
def dump_report():
    yield
    print("\n\nAll tests report:")
    print("\n".join(all_report_lines))

    stats_str = (
        "Tests passed: "
        + str(tests_stats["passed"])
        + ", failed: "
        + str(tests_stats["failed"])
        + ", skipped: "
        + str(tests_stats["skipped"])
        + "\n"
    )
    with open("all-tests-report.txt", "w") as f:
        f.write(stats_str + "\n")
        f.write("\n".join(all_report_lines))
    with open("failed-tests-report.txt", "w") as f:
        f.write(stats_str + "\n")
        f.write("\n".join(failed_report_lines))


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
