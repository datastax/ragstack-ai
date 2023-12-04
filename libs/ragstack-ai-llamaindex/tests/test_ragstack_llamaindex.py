import pytest


def test_import():
    from ragstack_ai_llamaindex import __version__  # noqa
    from llama_index.vector_stores import AstraDBVectorStore, CassandraVectorStore  # noqa
    import astrapy  # noqa
    import cassio  # noqa


def test_dont_download():
    # TODO: if you don't have the llama_index package installed, this test will fail because the monkey patch is not
    #  executed
    from ragstack_ai_llamaindex import __version__
    from llama_index import download_loader
    try:
        download_loader("PDFReader")
        pytest.fail("Should have raised an ImportError")
    except ImportError as error:
        assert "LlamaHub is disabled while using RAGStack AI" in str(error)
