import io
import tempfile
import uuid
from urllib.parse import urlparse

import boto3
from e2e_tests.conftest import (
    set_current_test_info,
)

from langchain.document_loaders import CSVLoader, WebBaseLoader, S3DirectoryLoader


def set_current_test_info_document_loader(doc_loader: str):
    set_current_test_info("langchain::document_loader", doc_loader)


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


def test_s3_loader():
    set_current_test_info_document_loader("s3")
    aws_region = "us-east-1"
    bucket_name = f"ragstack-ci-{uuid.uuid4()}"

    bucket = boto3.resource("s3", region_name=aws_region).Bucket(bucket_name)

    bucket.create(CreateBucketConfiguration={})
    s3_obj = bucket.Object("data.txt")

    try:
        data = io.BytesIO(b"test data")
        s3_obj.upload_fileobj(data)

        loader = S3DirectoryLoader(bucket_name, region_name=aws_region)
        docs = loader.load()

        for doc in docs:
            assert doc.page_content == "test data"
            assert urlparse(doc.metadata["source"]).path.lstrip("/") == "data.txt"
    finally:
        s3_obj.delete()
        bucket.delete()
