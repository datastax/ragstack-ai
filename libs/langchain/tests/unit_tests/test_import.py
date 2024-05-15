def test_import():
    import astrapy  # noqa
    import cassio  # noqa
    import langchain_core  # noqa
    import langsmith  # noqa
    import openai  # noqa
    import tiktoken  # noqa
    import unstructured_client  # noqa
    from langchain_astradb import AstraDBVectorStore  # noqa

    from langchain.vectorstores import AstraDB  # noqa


def test_meta():
    from importlib import metadata

    def check_meta(package: str):
        meta = metadata.metadata(package)
        assert meta["version"]
        assert meta["license"] == "BUSL-1.1"

    check_meta("ragstack-ai-langchain")
