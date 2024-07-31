from __future__ import annotations

from datetime import timedelta
import logging
from multiprocessing import Process
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
from threading import Thread
from typing import Optional

import humanize
from typing_extensions import Annotated
from typing_extensions import Doc


from trulens_eval.utils import notebook_utils
from trulens_eval.utils import python
from trulens_eval.utils.imports import static_resource

logger = logging.getLogger(__name__)

DASHBOARD_START_TIMEOUT: Annotated[int, Doc("Seconds to wait for dashboard to start")] = 30


def humanize_seconds(seconds: float):
    return humanize.naturaldelta(timedelta(seconds=seconds))


class Ragulate(python.SingletonPerName):
    """Ragulate is the main class that provides an entry points to ragulate.
    """

    _dashboard_urls: Optional[str] = None

    _dashboard_proc: Optional[Process] = None
    """[Process][multiprocessing.Process] executing the dashboard streamlit app.

    Is set to `None` if not executing.
    """

    _evaluator_stop: Optional[threading.Event] = None
    """Event for stopping the deferred evaluator which runs in another thread."""



    def find_unused_port(self) -> int:
        """Find an unused port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    def run_dashboard(
        self,
        port: Optional[int] = None,
        address: Optional[str] = None,
        force: bool = False,
        _dev: Optional[Path] = None
    ) -> Process:
        """Run a streamlit dashboard to view logged results and apps.

        Args:
           port: Port number to pass to streamlit through `server.port`.

           address: Address to pass to streamlit through `server.address`.

               **Address cannot be set if running from a colab
               notebook.**

           force: Stop existing dashboard(s) first. Defaults to `False`.

           _dev: If given, run dashboard with the given
              `PYTHONPATH`. This can be used to run the dashboard from outside
              of its pip package installation folder.

        Returns:
            The [Process][multiprocessing.Process] executing the streamlit
                dashboard.

        Raises:
            RuntimeError: Dashboard is already running. Can be avoided if `force`
                is set.

        """

        IN_COLAB = 'google.colab' in sys.modules
        if IN_COLAB and address is not None:
            raise ValueError("`address` argument cannot be used in colab.")

        if force:
            self.stop_dashboard(force=force)

        print("Starting dashboard ...")

        # Create .streamlit directory if it doesn't exist
        streamlit_dir = os.path.join(os.getcwd(), '.streamlit')
        os.makedirs(streamlit_dir, exist_ok=True)

        # Create config.toml file path
        config_path = os.path.join(streamlit_dir, 'config.toml')

        # Check if the file already exists
        if not os.path.exists(config_path):
            with open(config_path, 'w') as f:
                f.write('[theme]\n')
                f.write('primaryColor="#0A2C37"\n')
                f.write('backgroundColor="#FFFFFF"\n')
                f.write('secondaryBackgroundColor="F5F5F5"\n')
                f.write('textColor="#0A2C37"\n')
                f.write('font="sans serif"\n')
        else:
            print("Config file already exists. Skipping writing process.")

        # Create credentials.toml file path
        cred_path = os.path.join(streamlit_dir, 'credentials.toml')

        # Check if the file already exists
        if not os.path.exists(cred_path):
            with open(cred_path, 'w') as f:
                f.write('[general]\n')
                f.write('email=""\n')
        else:
            print("Credentials file already exists. Skipping writing process.")

        #run home with subprocess
        home_path = os.path.join("ragstack_ragulate","pages", "home.py")

        if Ragulate._dashboard_proc is not None:
            print("Dashboard already running at path:", Ragulate._dashboard_urls)
            return Ragulate._dashboard_proc

        env_opts = {}
        if _dev is not None:
            env_opts['env'] = os.environ
            env_opts['env']['PYTHONPATH'] = str(_dev)

        if port is None:
            port = self.find_unused_port()

        args = ["streamlit", "run", "--server.headless=True"]
        if port is not None:
            args.append(f"--server.port={port}")
        if address is not None:
            args.append(f"--server.address={address}")

        args += [home_path]

        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **env_opts
        )

        started = threading.Event()
        tunnel_started = threading.Event()
        if notebook_utils.is_notebook():
            out_stdout, out_stderr = notebook_utils.setup_widget_stdout_stderr()
        else:
            out_stdout = None
            out_stderr = None

        if IN_COLAB:
            tunnel_proc = subprocess.Popen(
                ["npx", "localtunnel", "--port",
                 str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                **env_opts
            )

            def listen_to_tunnel(proc: subprocess.Popen, pipe, out, started):
                while proc.poll() is None:

                    line = pipe.readline()
                    if "url" in line:
                        started.set()
                        line = "Go to this url and submit the ip given here. " + line

                    if out is not None:
                        out.append_stdout(line)

                    else:
                        print(line)

            Ragulate.tunnel_listener_stdout = Thread(
                target=listen_to_tunnel,
                args=(
                    tunnel_proc, tunnel_proc.stdout, out_stdout, tunnel_started
                )
            )
            Ragulate.tunnel_listener_stderr = Thread(
                target=listen_to_tunnel,
                args=(
                    tunnel_proc, tunnel_proc.stderr, out_stderr, tunnel_started
                )
            )
            Ragulate.tunnel_listener_stdout.daemon = True
            Ragulate.tunnel_listener_stderr.daemon = True
            Ragulate.tunnel_listener_stdout.start()
            Ragulate.tunnel_listener_stderr.start()
            if not tunnel_started.wait(timeout=DASHBOARD_START_TIMEOUT
                                      ):  # This might not work on windows.
                raise RuntimeError("Tunnel failed to start in time. ")

        def listen_to_dashboard(proc: subprocess.Popen, pipe, out, started):
            while proc.poll() is None:
                line = pipe.readline()
                if IN_COLAB:
                    if "External URL: " in line:
                        started.set()
                        line = line.replace(
                            "External URL: http://", "Submit this IP Address: "
                        )
                        line = line.replace(f":{port}", "")
                        if out is not None:
                            out.append_stdout(line)
                        else:
                            print(line)
                        Ragulate._dashboard_urls = line  # store the url when dashboard is started
                else:
                    if "Network URL: " in line:
                        url = line.split(": ")[1]
                        url = url.rstrip()
                        print(f"Dashboard started at {url} .")
                        started.set()
                        Ragulate._dashboard_urls = line  # store the url when dashboard is started
                    if out is not None:
                        out.append_stdout(line)
                    else:
                        print(line)
            if out is not None:
                out.append_stdout("Dashboard closed.")
            else:
                print("Dashboard closed.")

        Ragulate.dashboard_listener_stdout = Thread(
            target=listen_to_dashboard,
            args=(proc, proc.stdout, out_stdout, started)
        )
        Ragulate.dashboard_listener_stderr = Thread(
            target=listen_to_dashboard,
            args=(proc, proc.stderr, out_stderr, started)
        )

        # Purposely block main process from ending and wait for dashboard.
        Ragulate.dashboard_listener_stdout.daemon = False
        Ragulate.dashboard_listener_stderr.daemon = False

        Ragulate.dashboard_listener_stdout.start()
        Ragulate.dashboard_listener_stderr.start()

        Ragulate._dashboard_proc = proc

        wait_period = DASHBOARD_START_TIMEOUT
        if IN_COLAB:
            # Need more time to setup 2 processes tunnel and dashboard
            wait_period = wait_period * 3

        # This might not work on windows.
        if not started.wait(timeout=wait_period):
            Ragulate._dashboard_proc = None
            raise RuntimeError(
                "Dashboard failed to start in time. "
                "Please inspect dashboard logs for additional information."
            )

        return proc

    start_dashboard = run_dashboard

    def stop_dashboard(self, force: bool = False) -> None:
        """
        Stop existing dashboard(s) if running.

        Args:
            force: Also try to find any other dashboard processes not
                started in this notebook and shut them down too.

                **This option is not supported under windows.**

        Raises:
             RuntimeError: Dashboard is not running in the current process. Can be avoided with `force`.
        """
        if Ragulate._dashboard_proc is None:
            if not force:
                raise RuntimeError(
                    "Dashboard not running in this workspace. "
                    "You may be able to shut other instances by setting the `force` flag."
                )

            else:
                if sys.platform.startswith("win"):
                    raise RuntimeError(
                        "Force stop option is not supported on windows."
                    )

                print("Force stopping dashboard ...")
                import os
                import pwd  # PROBLEM: does not exist on windows

                import psutil
                username = pwd.getpwuid(os.getuid())[0]
                for p in psutil.process_iter():
                    try:
                        cmd = " ".join(p.cmdline())
                        if "streamlit" in cmd and "Leaderboard.py" in cmd and p.username(
                        ) == username:
                            print(f"killing {p}")
                            p.kill()
                    except Exception as e:
                        continue

        else:
            Ragulate._dashboard_proc.kill()
            Ragulate._dashboard_proc = None

if __name__ == "__main__":
    ragulate = Ragulate()
    ragulate.run_dashboard(port=8000)