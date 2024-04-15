from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs


class CassandraContainer(DockerContainer):
    def __init__(
        self,
        image: str = "docker.io/stargateio/dse-next:4.0.11-b259738f492f",
        port: int = 9042,
        **kwargs,
    ) -> None:
        super(CassandraContainer, self).__init__(image=image, **kwargs)
        self.port = port

        self.with_exposed_ports(self.port)

    def _configure(self):
        pass

    def start(self):
        start_res = super().start()
        wait_for_logs(self, "Startup complete")
        return start_res

    def get_mapped_port(self):
        return self.get_exposed_port(self.port)
