def test_import():
    from ragstack_ai_llamaindex import __version__  # noqa
    from llama_index.vector_stores import AstraDBVectorStore, CassandraVectorStore  # noqa
    import astrapy  # noqa
    import cassio  # noqa
