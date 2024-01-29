import logging
import os
import pathlib
import time

import pytest


from e2e_tests.test_utils.astradb_vector_store_handler import AstraDBVectorStoreHandler
from e2e_tests.test_utils.cassandra_vector_store_handler import (
    CassandraVectorStoreHandler,
)
from e2e_tests.test_utils.vector_store_handler import (
    VectorStoreHandler,
    VectorStoreImplementation,
)
from e2e_tests.test_utils import (
    get_required_env as root_get_required_env,
    is_skipped_due_to_implementation_not_supported,
)

LOGGER = logging.getLogger(__name__)

DIR_PATH = os.path.dirname(os.path.abspath(__file__))


# Loading the .env file if it exists
def _load_env() -> None:
    dotenv_path = os.path.join(DIR_PATH, os.pardir, ".env")
    if os.path.exists(dotenv_path):
        from dotenv import load_dotenv

        load_dotenv(dotenv_path)


_load_env()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


# Uncomment to enable debug logging on Astra calls
# logging.getLogger('astrapy.utils').setLevel(logging.DEBUG)


def get_required_env(name) -> str:
    return root_get_required_env(name)


vector_database_type = os.environ.get("VECTOR_DATABASE_TYPE", "astradb")
if vector_database_type not in ["astradb", "local-cassandra"]:
    raise ValueError(f"Invalid VECTOR_DATABASE_TYPE: {vector_database_type}")

is_astra = vector_database_type == "astradb"


def get_vector_store_handler(
    implementation: VectorStoreImplementation,
) -> VectorStoreHandler:
    if vector_database_type == "astradb":
        return AstraDBVectorStoreHandler(implementation)
    elif vector_database_type == "local-cassandra":
        return CassandraVectorStoreHandler(implementation)


failed_report_lines = []
all_report_lines = []
langchain_report_lines = []
llamaindex_report_lines = []
tests_stats = {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
}


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
        skip_report_line = (
            rep.outcome == "skipped"
            and is_skipped_due_to_implementation_not_supported(result)
        )
        if not skip_report_line:
            logging.info("Test report line: " + report_line)
            if rep.outcome != "passed":
                # also keep skipped tests in the report
                failed_report_lines.append(report_line)
            all_report_lines.append(report_line)
            if is_langchain:
                langchain_report_lines.append(report_line)
            elif is_llamaindex:
                llamaindex_report_lines.append(report_line)
        else:
            logging.info("Skipping test report line: " + result)
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


# azure-open-ai
os.environ["AZURE_OPEN_AI_CHAT_MODEL_DEPLOYMENT"] = "gpt-35-turbo"
os.environ["AZURE_OPEN_AI_EMBEDDINGS_MODEL_DEPLOYMENT"] = "text-embedding-ada-002"

# vertex-ai
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    with open("/tmp/gcloud-account-key.json", "w") as f:
        f.write(os.getenv("GCLOUD_ACCOUNT_KEY_JSON", ""))
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcloud-account-key.json"
