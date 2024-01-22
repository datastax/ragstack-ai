import logging
import os
import pathlib
import random
import time
import uuid
from dataclasses import dataclass

import cassio
import pytest
from astrapy.db import AstraDB as LibAstraDB
from cassandra.cluster import Cluster, PlainTextAuthProvider, Session

from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs


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
    value = os.environ[name]
    if not value:
        LOGGER.warning(f"Empty required environment variable: {name}")
        pytest.skip(f"Empty required environment variable: {name}")
    return value


@dataclass
class AstraRef:
    token: str
    api_endpoint: str
    collection: str
    id: str
    env: str


class VectorDatabaseHandler:
    def __init__(self):
        self.mode = os.environ.get("VECTOR_DATABASE_MODE", "astra")
        if self.mode not in ["astra", "dse"]:
            raise ValueError(f"Invalid VECTOR_DATABASE_MODE: {self.mode}")
        self.test_table_name = None

    def is_astradb(self) -> bool:
        return self.mode == "astra"

    def is_dse(self) -> bool:
        return self.mode == "dse"

    def get_astra_ref(self) -> AstraRef:
        if not self.is_astradb():
            raise ValueError("Not an AstraDB test")
        env = os.environ.get("ASTRA_DB_ENV", "prod").lower()
        return AstraRef(
            token=get_required_env("ASTRA_DB_TOKEN"),
            api_endpoint=get_required_env("ASTRA_DB_ENDPOINT"),
            collection=get_required_env("ASTRA_TABLE_NAME"),
            id=get_required_env("ASTRA_DB_ID"),
            env=env,
        )

    def get_astra_client(self) -> LibAstraDB:
        if not self.is_astradb():
            raise ValueError("Not an AstraDB test")
        astra_ref = self.get_astra_ref()
        return LibAstraDB(
            token=astra_ref.token,
            api_endpoint=astra_ref.api_endpoint,
        )

    def get_cassandra_session(self) -> Session:
        if not self.is_dse():
            raise ValueError("Not running in dse mode")
        return self.cassandra_session

    def after_test(self):
        self.test_table_name = None
        if self.mode == "astra":
            astra_ref = self.get_astra_ref()
            delete_astra_collection(astra_ref)
            delete_all_astra_collections_with_client(
                LibAstraDB(
                    token=astra_ref.token,
                    api_endpoint=astra_ref.api_endpoint,
                )
            )
        pass

    def before_test(self, implementation):
        if self.mode == "astra":
            self.test_table_name = get_required_env("ASTRA_TABLE_NAME")
            astra_ref = self.get_astra_ref()
            astra_db_client = LibAstraDB(
                token=astra_ref.token,
                api_endpoint=astra_ref.api_endpoint,
            )
            delete_all_astra_collections_with_client(astra_db_client)

            if implementation == "cassandra":
                # to run cassandra implementation over astra
                if astra_ref.env == "dev":
                    bundle_url_template = "https://api.dev.cloud.datastax.com/v2/databases/{database_id}/secureBundleURL"
                    cassio.init(
                        token=astra_ref.token,
                        database_id=astra_ref.id,
                        bundle_url_template=bundle_url_template,
                    )
                else:
                    cassio.init(token=astra_ref.token, database_id=astra_ref.id)
        elif self.mode == "dse":
            self.test_table_name = "table_" + str(random.randint(0, 1000000))
            cassandra_container_ref = []
            if len(cassandra_container_ref) == 0:
                cassandra_container = CassandraContainer()
                cassandra_container.start()
                logging.info("Cassandra container started")
                cassandra_container_ref.append(cassandra_container)
            else:
                logging.info("Cassandra container already started")

            self.cassandra_session = cassandra_container_ref[0].create_session()
            cassio.init(session=self.cassandra_session)

    def get_table_name(self) -> str:
        return self.test_table_name


vector_database_handler = VectorDatabaseHandler()


def get_vector_database_handler() -> VectorDatabaseHandler:
    return vector_database_handler


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


class CassandraContainer(DockerContainer):
    def __init__(
        self,
        image: str = "docker.io/stargateio/dse-next:4.0.11-b259738f492f",
        port: int = 9042,
        keyspace: str = "default_keyspace",
        **kwargs,
    ) -> None:
        super(CassandraContainer, self).__init__(image=image, **kwargs)
        self.keyspace = keyspace
        self.port = port

        self.with_exposed_ports(self.port)

    def _configure(self):
        pass

    def start(self):
        start_res = super().start()
        wait_for_logs(self, "Startup complete")
        return start_res

    def create_session(self) -> Session:
        actual_port = self.get_exposed_port(self.port)
        cluster = Cluster(
            [("127.0.0.1", actual_port)],
            auth_provider=PlainTextAuthProvider("cassandra", "cassandra"),
        )
        session = cluster.connect()
        session.execute(f"DROP KEYSPACE IF EXISTS {self.keyspace}")
        session.execute(
            f"CREATE KEYSPACE IF NOT EXISTS {self.keyspace} WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': '1'}}"
        )
        return session


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
