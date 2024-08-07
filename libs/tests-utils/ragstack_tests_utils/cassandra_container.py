from typing import Any

from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs
from typing_extensions import Self


class CassandraContainer(DockerContainer):  # type: ignore[misc]
    def __init__(
        self,
        image: str = "docker.io/stargateio/dse-next:4.0.11-b259738f492f",
        port: int = 9042,
        **kwargs: Any,
    ) -> None:
        super().__init__(image=image, **kwargs)
        self.port = port

        self.with_env(
            "JVM_OPTS",
            "-Dcassandra.skip_wait_for_gossip_to_settle=0 -Dcassandra.initial_token=0",
        )
        self.with_env("HEAP_NEWSIZE", "128M")
        self.with_env("MAX_HEAP_SIZE", "1024M")
        self.with_env("CASSANDRA_ENDPOINT_SNITCH", "GossipingPropertyFileSnitch")
        self.with_env("CASSANDRA_DC", "datacenter1")
        self.with_exposed_ports(self.port)

    def start(self) -> Self:
        super().start()
        wait_for_logs(self, "Startup complete")
        return self

    def get_mapped_port(self) -> int:
        return int(self.get_exposed_port(self.port))
