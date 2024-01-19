import logging
import os
import pathlib
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
    env: str


def get_astra_ref() -> AstraRef:
    env = os.environ.get("ASTRA_DB_ENV", "prod").lower()
    return AstraRef(
        token=get_required_env("ASTRA_DB_TOKEN"),
        api_endpoint=get_required_env("ASTRA_DB_ENDPOINT"),
        collection=get_required_env("ASTRA_TABLE_NAME"),
        id=get_required_env("ASTRA_DB_ID"),
        env=env,
    )


def delete_all_astra_collections_with_client(raw_client: LibAstraDB):
    """
    Deletes all collections.

    Current AstraDB has a limit of 5 collections, meaning orphaned collections
    will cause subsequent tests to fail if the limit is reached.
    """
    collections = raw_client.get_collections().get("status").get("collections")
    logging.info(f"Existing collections: {collections}")
    for collection_name in collections:
        truncate_and_delete(raw_client, collection_name)


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
    collection_name = astra_ref.collection
    truncate_and_delete(raw_client, collection_name)


failed_report_lines = []
all_report_lines = []
langchain_report_lines = []
llamaindex_report_lines = []
tests_stats = {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
}


def truncate_and_delete(client: LibAstraDB, collection: str):
    """
    Truncates all documents from the collection, then deletes the collection.

    Attempts to alleviate the load from delete_collection, which currently experiences
    a high rate of timeouts.
    """
    logging.info(f"Deleting collection: {collection}")
    astra_db_collection = client.collection(collection_name=collection)
    astra_db_collection.delete_many(filter={})

    response = client.delete_collection(collection_name=collection)
    logging.info(f"Delete collection response: {response}")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    # also get the setup phase if failed
    if rep.outcome != "passed" or rep.when == "call":
        if (
            "RAGSTACK_E2E_TESTS_TEST_START" not in os.environ
            or not os.environ["RAGSTACK_E2E_TESTS_TEST_START"]
        ):
            total_time = "?"
        else:
            start_time = int(os.environ["RAGSTACK_E2E_TESTS_TEST_START"])
            total_time = round((time.perf_counter_ns() - start_time) / 1e9)

        os.environ["RAGSTACK_E2E_TESTS_TEST_START"] = ""

        info = os.getenv("RAGSTACK_E2E_TESTS_TEST_INFO", "")
        if not info:
            test_path = pathlib.PurePath(item.path)
            info = test_path.parent.name + "::" + test_path.name + "::" + item.name
        logging.info(f"Test {info} took: {total_time} seconds")
        paths = str(item.path).split(os.sep)
        is_langchain = False
        is_llamaindex = False
        if "langchain" in paths:
            is_langchain = True
        elif "llama_index" in paths:
            is_llamaindex = True

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
        if is_langchain:
            langchain_report_lines.append(report_line)
        elif is_llamaindex:
            llamaindex_report_lines.append(report_line)
        os.environ["RAGSTACK_E2E_TESTS_TEST_INFO"] = ""

    if rep.when == "call":
        os.environ["RAGSTACK_E2E_TESTS_TEST_START"] = str(time.perf_counter_ns())


def set_current_test_info(test_name: str, test_info: str):
    test_info = test_info.replace("_", "-")
    os.environ["RAGSTACK_E2E_TESTS_TEST_INFO"] = f"{test_name}::{test_info}"


@pytest.fixture(scope="session", autouse=True)
def dump_report():
    yield
    logging.info("All tests report:")
    logging.info("\n".join(all_report_lines))
    logging.info("Failed tests report:")
    logging.info("\n".join(failed_report_lines))

    stats_str = (
        "Tests passed: "
        + str(tests_stats["passed"])
        + ", failed: "
        + str(tests_stats["failed"])
        + ", skipped: "
        + str(tests_stats["skipped"])
        + "\n"
    )
    _report_to_file(stats_str, "all-tests-report.txt", all_report_lines)
    _report_to_file(stats_str, "failed-tests-report.txt", failed_report_lines)

    _report_to_file("", "langchain-tests-report.txt", langchain_report_lines)
    _report_to_file("", "llamaindex-tests-report.txt", llamaindex_report_lines)


def _report_to_file(stats_str: str, filename: str, report_lines: list):
    report_lines.sort()
    with open(filename, "w") as f:
        if stats_str:
            f.write(stats_str + "\n")
        f.write("\n".join(report_lines))


# astra
os.environ["ASTRA_TABLE_NAME"] = f"documents_{random_string()}"

# azure-open-ai
os.environ["AZURE_OPEN_AI_CHAT_MODEL_DEPLOYMENT"] = "gpt-35-turbo"
os.environ["AZURE_OPEN_AI_EMBEDDINGS_MODEL_DEPLOYMENT"] = "text-embedding-ada-002"

# vertex-ai
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    with open("/tmp/gcloud-account-key.json", "w") as f:
        f.write(os.getenv("GCLOUD_ACCOUNT_KEY_JSON", ""))
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcloud-account-key.json"
