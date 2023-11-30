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


report_lines = []


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call":
        info = os.getenv("RAGSTACK_E2E_TESTS_TEST_INFO", "")
        if not info:
            info = os.path.basename(item.path) + "::" + item.name
        test_outcome = "❌" if rep.failed else "✅"
        result = " " + str(call.excinfo) if call.excinfo else ""
        report_line = f"{info} -> {test_outcome}{result}"
        report_lines.append(report_line)


def set_current_test_info_simple_rag(llm: str, embedding: str, vector_db: str) -> None:
    os.environ[
        "RAGSTACK_E2E_TESTS_TEST_INFO"
    ] = f"simple_rag::{llm},{embedding},{vector_db}"


def set_current_test_info_document_loader(doc_loader: str) -> None:
    os.environ["RAGSTACK_E2E_TESTS_TEST_INFO"] = f"document_loader::{doc_loader}"


@pytest.fixture(scope="session", autouse=True)
def dump_report():
    yield
    print("\n\nTest report:")
    print("\n".join(report_lines))
    with open("generated-test-report.txt", "w") as f:
        f.write("\n".join(report_lines))


# astra dev
os.environ["ASTRA_DEV_TABLE_NAME"] = f"documents_{random_string()}"

# astra prod
os.environ["ASTRA_PROD_TABLE_NAME"] = f"documents_{random_string()}"
os.environ["ASTRA_PROD_KEYSPACE"] = "default_keyspace"

# azure-open-ai
os.environ["AZURE_OPEN_AI_CHAT_MODEL_DEPLOYMENT"] = "gpt-35-turbo"
os.environ["AZURE_OPEN_AI_EMBEDDINGS_MODEL_DEPLOYMENT"] = "text-embedding-ada-002"

# vertex-ai
with open("/tmp/gcloud-account-key.json", "w") as f:
    f.write(os.getenv("GCLOUD_ACCOUNT_KEY_JSON", ""))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcloud-account-key.json"
