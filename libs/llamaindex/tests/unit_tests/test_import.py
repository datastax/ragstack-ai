import importlib
from typing import Any, Callable


def test_import() -> None:
    import astrapy  # noqa: F401
    import cassio  # noqa: F401
    import openai  # noqa: F401
    import tiktoken  # noqa: F401
    import unstructured  # noqa: F401
    from llama_index.vector_stores.astra_db import AstraDBVectorStore  # noqa: F401
    from llama_index.vector_stores.cassandra import CassandraVectorStore  # noqa: F401


def check_no_import(fn: Callable[[], Any]) -> None:
    try:
        fn()
        msg = "Should have failed to import"
        raise RuntimeError(msg)
    except ImportError:
        pass


def test_not_import() -> None:
    check_no_import(lambda: importlib.import_module("langchain.vectorstores"))
    check_no_import(lambda: importlib.import_module("langchain_astradb"))
    check_no_import(lambda: importlib.import_module("langchain_core"))
    check_no_import(lambda: importlib.import_module("langsmith"))


def test_meta() -> None:
    from importlib import metadata

    def check_meta(package: str) -> None:
        meta = metadata.metadata(package)
        assert meta["version"]
        assert meta["license"] == "BUSL-1.1"

    check_meta("ragstack-ai-llamaindex")
