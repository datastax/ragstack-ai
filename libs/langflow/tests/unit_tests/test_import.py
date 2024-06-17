import importlib


def test_import():
    import astrapy  # noqa
    import cassio  # noqa
    import openai  # noqa
    import tiktoken  # noqa
    import unstructured  # noqa
    # TODO: uncomment once it works
    #from langflow.components.vectorstores.AstraDB import AstraDBVectorStoreComponent # noqa


def test_meta():
    from importlib import metadata
    def check_meta(package: str):
        meta = metadata.metadata(package)
        assert meta["version"]
        assert meta["license"] == "BUSL-1.1"

    check_meta("ragstack-ai-langflow")
