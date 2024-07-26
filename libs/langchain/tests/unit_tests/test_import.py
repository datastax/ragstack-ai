def test_import() -> None:
    import astrapy  # noqa: F401
    import cassio  # noqa: F401
    import langchain_core  # noqa: F401
    import langsmith  # noqa: F401
    import openai  # noqa: F401
    import tiktoken  # noqa: F401
    import unstructured  # noqa: F401
    from langchain.vectorstores import AstraDB  # noqa: F401
    from langchain_astradb import AstraDBVectorStore  # noqa: F401


def test_meta() -> None:
    from importlib import metadata

    def check_meta(package: str) -> None:
        meta = metadata.metadata(package)
        assert meta["version"]
        assert meta["license"] == "BUSL-1.1"

    check_meta("ragstack-ai-langchain")
