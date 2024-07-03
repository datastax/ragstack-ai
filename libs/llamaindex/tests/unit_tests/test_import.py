import importlib


def test_import():
    import astrapy  # noqa
    import cassio  # noqa
    import openai  # noqa
    import tiktoken  # noqa
    import unstructured  # noqa
    from llama_index.vector_stores.astra_db import AstraDBVectorStore  # noqa
    from llama_index.vector_stores.cassandra import CassandraVectorStore  # noqa


def check_no_import(fn: callable):
    try:
        fn()
        raise RuntimeError("Should have failed to import")
    except ImportError:
        pass


def test_not_import():
    check_no_import(lambda: importlib.import_module("langchain.vectorstores"))
    check_no_import(lambda: importlib.import_module("langchain_astradb"))
    check_no_import(lambda: importlib.import_module("langchain_core"))
    check_no_import(lambda: importlib.import_module("langsmith"))


def test_meta():
    from importlib import metadata

    def check_meta(package: str):
        meta = metadata.metadata(package)
        assert meta["version"]
        assert meta["license"] == "BUSL-1.1"

    check_meta("ragstack-ai-llamaindex")
