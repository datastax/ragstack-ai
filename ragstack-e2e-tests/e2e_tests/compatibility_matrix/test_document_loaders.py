import tempfile

from e2e_tests.compatibility_matrix.conftest import (
    set_current_test_info_document_loader,
)

from langchain.document_loaders import CSVLoader, WebBaseLoader


def test_csv_loader():
    set_current_test_info_document_loader("csv")
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv") as temp_csv_file:
        with open(temp_csv_file.name, "w") as write:
            write.writelines(
                [
                    "column1,column2,column3\n",
                    "value1,value2,value3\n",
                    "value4,value5,value6\n",
                    "value7,value8,value9\n",
                ]
            )
        docs = CSVLoader(file_path=temp_csv_file.name).load()
        assert len(docs) == 3

        doc1 = docs[0]
        assert doc1.page_content == "column1: value1\ncolumn2: value2\ncolumn3: value3"
        assert doc1.metadata == {"row": 0, "source": temp_csv_file.name}


def test_web_based_loader():
    set_current_test_info_document_loader("web")
    loader = WebBaseLoader(
        ["https://langstream.ai/changelog/", "https://langstream.ai/faq/"]
    )
    loader.requests_per_second = 1
    docs = loader.load()
    assert len(docs) == 2

    doc1 = docs[0]
    assert "0.1.0 - Oct 4, 2023" in doc1.page_content
    assert doc1.metadata == {
        "source": "https://langstream.ai/changelog/",
        "title": "LangStream Changelog | LangStream: an Event-Driven Developer Platform for LLM Apps",  # noqa: E501
        "description": "Notes from releases",
        "language": "en",
    }

    doc2 = docs[1]
    assert "LangStream is an open-source project" in doc2.page_content
    assert doc2.metadata == {
        "source": "https://langstream.ai/faq/",
        "title": "FAQ | LangStream: an Event-Driven Developer Platform for LLM Apps",
        "description": "Frequently Asked Questions",
        "language": "en",
    }
