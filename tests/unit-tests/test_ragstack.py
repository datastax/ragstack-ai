def test_import():
    from llama_index.vector_stores import (
        AstraDBVectorStore,
        CassandraVectorStore,
    )  # noqa
    from langchain.vectorstores import AstraDB  # noqa
    import langchain_core  # noqa
    import langsmith  # noqa
    import astrapy  # noqa
    import cassio  # noqa
    import unstructured  # noqa
    import openai  # noqa
    import tiktoken  # noqa
