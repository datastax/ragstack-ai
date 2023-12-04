def test_import():
    from llama_index.vector_stores import AstraDBVectorStore, CassandraVectorStore  # noqa
    import astrapy  # noqa
    import cassio  # noqa
