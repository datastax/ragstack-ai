import logging
import os
import time
import uuid
from dataclasses import dataclass

import pytest
from astrapy.db import AstraDB as LibAstraDB

LOGGER = logging.getLogger(__name__)


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
        LOGGER.warning(f"Missing required environment variable: {name}")
        pytest.skip(f"Missing required environment variable: {name}")
    return os.environ[name]


@dataclass
class AstraRef:
    token: str
    api_endpoint: str
    collection: str
    id: str


def get_astra_ref() -> AstraRef:
    return AstraRef(
        token=get_required_env("ASTRA_DB_TOKEN"),
        api_endpoint=get_required_env("ASTRA_DB_ENDPOINT"),
        collection=get_required_env("ASTRA_TABLE_NAME"),
        id=get_required_env("ASTRA_DB_ID"),
    )


def delete_all_astra_collections_with_client(raw_client: LibAstraDB):
    """
    Deletes all collections.

    Current AstraDB has a limit of 5 collections, meaning orphaned collections
    will cause subsequent tests to fail if the limit is reached.
    """
    collections = raw_client.get_collections().get("status").get("collections")
    logging.info(f"Existing collections: {collections}")
    for collection_info in collections:
        logging.info(f"Deleting collection: {collection_info}")
        raw_client.delete_collection(collection_info)


def delete_all_astra_collections(astra_ref: AstraRef):
    """
    Deletes all collections.

    Current AstraDB has a limit of 5 collections, meaning orphaned collections
    will cause subsequent tests to fail if the limit is reached.
    """
    raw_client = LibAstraDB(api_endpoint=astra_ref.api_endpoint, token=astra_ref.token)
    delete_all_astra_collections_with_client(raw_client)


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
    info = os.path.basename(item.path) + "::" + item.name

    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call":
        start_time = int(os.environ["RAGSTACK_E2E_TESTS_TEST_START"])
        os.environ["RAGSTACK_E2E_TESTS_TEST_START"] = ""
        total_time = round((time.perf_counter_ns() - start_time) / 1e9)
        logging.info(f"Test {info} took: {total_time} seconds")
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
        report_line = f"{info} -> {test_outcome}{result} ({total_time} s)"
        logging.info("Test report line: " + report_line)
        if rep.outcome != "passed":
            # also keep skipped tests in the report
            failed_report_lines.append(report_line)
        all_report_lines.append(report_line)
        os.environ["RAGSTACK_E2E_TESTS_TEST_INFO"] = ""
    else:
        os.environ["RAGSTACK_E2E_TESTS_TEST_START"] = str(time.perf_counter_ns())


def set_current_test_info(test_name: str, test_info: str):
    os.environ["RAGSTACK_E2E_TESTS_TEST_INFO"] = f"{test_name}::{test_info}"


@pytest.fixture(scope="session", autouse=True)
def dump_report():
    yield
    logging.info("All tests report:")
    logging.info("\n".join(all_report_lines))

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


# astra
os.environ["ASTRA_TABLE_NAME"] = f"documents_{random_string()}"

# azure-open-ai
os.environ["AZURE_OPEN_AI_CHAT_MODEL_DEPLOYMENT"] = "gpt-35-turbo"
os.environ["AZURE_OPEN_AI_EMBEDDINGS_MODEL_DEPLOYMENT"] = "text-embedding-ada-002"

# vertex-ai
with open("/tmp/gcloud-account-key.json", "w") as f:
    f.write(os.getenv("GCLOUD_ACCOUNT_KEY_JSON", ""))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcloud-account-key.json"
