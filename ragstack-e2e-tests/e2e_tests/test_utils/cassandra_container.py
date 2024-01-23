from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Session, Cluster
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs


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
