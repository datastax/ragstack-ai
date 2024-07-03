import logging
import time
from typing import Callable

from langsmith import Client

LANGSMITH_CLIENT = Client()


def record_langsmith_sharelink(
    run_id: str, record_property: Callable, tries: int = 6
) -> None:
    try:
        sharelink = LANGSMITH_CLIENT.share_run(run_id)
        record_property("langsmith_url", sharelink)
        logging.info(f"recorded langsmith link: {sharelink}")
    except Exception:
        # runs may take a while to be discoverable
        if tries < 0:
            raise
        time.sleep(5)
        record_langsmith_sharelink(run_id, record_property, tries - 1)
