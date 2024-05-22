import secrets
from typing import Iterator, List

import pytest
from cassandra.cluster import Cluster, Session
from langchain.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs

from knowledge_graph.cassandra_graph_store import CassandraGraphStore


@pytest.fixture(scope="session")
def db_keyspace() -> str:
    return "default_keyspace"


@pytest.fixture(scope="session")
def cassandra_port(db_keyspace: str) -> Iterator[int]:
    # TODO: Allow running against local Cassandra and/or Astra using pytest option.
    cassandra = DockerContainer("cassandra:5")
    cassandra.with_exposed_ports(9042)
    cassandra.with_env(
        "JVM_OPTS",
        "-Dcassandra.skip_wait_for_gossip_to_settle=0 -Dcassandra.initial_token=0",
    )
    cassandra.with_env("HEAP_NEWSIZE", "128M")
    cassandra.with_env("MAX_HEAP_SIZE", "1024M")
    cassandra.with_env("CASSANDRA_ENDPOINT_SNITCH", "GossipingPropertyFileSnitch")
    cassandra.with_env("CASSANDRA_DC", "datacenter1")
    cassandra.start()
    wait_for_logs(cassandra, "Startup complete")
    cassandra.get_wrapped_container().exec_run(
        (
            f"""cqlsh -e "CREATE KEYSPACE {db_keyspace} WITH replication = """
            '''{'class': 'SimpleStrategy', 'replication_factor': '1'};"'''
        )
    )
    port = cassandra.get_exposed_port(9042)
    print(f"Cassandra started. Port is {port}")
    yield port
    cassandra.stop()


@pytest.fixture(scope="session")
def db_session(cassandra_port: int) -> Session:
    print(f"Connecting to cassandra on {cassandra_port}")
    cluster = Cluster(
        port=cassandra_port,
    )
    return cluster.connect()


@pytest.fixture(scope="session")
def llm() -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model_name="gpt-4-turbo-2024-04-09", temperature=0.0)
        return model
    except ValueError:
        pytest.skip("Unable to create OpenAI model")


class DataFixture:
    def __init__(self, session: Session, keyspace: str, documents: List[GraphDocument]) -> None:
        self.session = session
        self.keyspace = "default_keyspace"
        self.uid = secrets.token_hex(8)
        self.node_table = f"entities_{self.uid}"
        self.edge_table = f"relationships_{self.uid}"

        text_embeddings = None
        try:
            from langchain_openai import OpenAIEmbeddings

            text_embeddings = OpenAIEmbeddings()
        except ValueError:
            print("OpenAI not configured. Not embedding data.")
        self.has_embeddings = text_embeddings is not None

        self.graph_store = CassandraGraphStore(
            node_table=self.node_table,
            edge_table=self.edge_table,
            text_embeddings=text_embeddings,
            session=session,
            keyspace=keyspace,
        )

        self.graph_store.add_graph_documents(documents)

    def drop(self):
        self.session.execute(f"DROP TABLE IF EXISTS {self.keyspace}.{self.node_table};")
        self.session.execute(f"DROP TABLE IF EXISTS {self.keyspace}.{self.edge_table};")


@pytest.fixture(scope="session")
def marie_curie(db_session: Session, db_keyspace: str) -> Iterator[DataFixture]:
    marie_curie = Node(id="Marie Curie", type="Person")
    pierre_curie = Node(id="Pierre Curie", type="Person")
    nobel_prize = Node(id="Nobel Prize", type="Award")
    university_of_paris = Node(id="University of Paris", type="Organization")
    polish = Node(id="Polish", type="Nationality", properties={"European": True})
    french = Node(id="French", type="Nationality", properties={"European": True})
    physicist = Node(id="Physicist", type="Profession")
    chemist = Node(id="Chemist", type="Profession")
    radioactivity = Node(id="Radioactivity", type="Scientific concept")
    professor = Node(id="Professor", type="Profession")
    document = GraphDocument(
        nodes=[
            marie_curie,
            pierre_curie,
            nobel_prize,
            university_of_paris,
            polish,
            french,
            physicist,
            chemist,
            radioactivity,
            professor,
        ],
        relationships=[
            Relationship(source=marie_curie, target=polish, type="HAS_NATIONALITY"),
            Relationship(source=marie_curie, target=french, type="HAS_NATIONALITY"),
            Relationship(source=marie_curie, target=physicist, type="HAS_PROFESSION"),
            Relationship(source=marie_curie, target=chemist, type="HAS_PROFESSION"),
            Relationship(source=marie_curie, target=radioactivity, type="RESEARCHED"),
            Relationship(source=marie_curie, target=nobel_prize, type="WON"),
            Relationship(source=pierre_curie, target=nobel_prize, type="WON"),
            Relationship(source=marie_curie, target=pierre_curie, type="MARRIED_TO"),
            Relationship(source=marie_curie, target=university_of_paris, type="WORKED_AT"),
            Relationship(source=marie_curie, target=professor, type="HAS_PROFESSION"),
        ],
        source=Document(page_content="test_content"),
    )
    data = DataFixture(session=db_session, keyspace=db_keyspace, documents=[document])
    yield data
    data.drop()
