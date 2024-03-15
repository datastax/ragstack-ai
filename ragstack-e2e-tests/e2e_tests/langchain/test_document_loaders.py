import io
import json
import os
import tempfile
import uuid
from urllib.parse import urlparse

import boto3
import pytest
from azure.storage.blob import ContainerClient
from langchain_astradb import AstraDBLoader

from e2e_tests.conftest import set_current_test_info, get_required_env, is_astra

from langchain.document_loaders import CSVLoader, WebBaseLoader, S3DirectoryLoader
from langchain_community.document_loaders import AzureBlobStorageContainerLoader

from e2e_tests.test_utils.astradb_vector_store_handler import AstraDBVectorStoreHandler
from e2e_tests.test_utils.vector_store_handler import VectorStoreImplementation


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

    bucket = boto3.resource("s3", region_name=aws_region).create_bucket(
        Bucket=bucket_name
    )

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


def test_azure_blob_doc_loader():
    set_current_test_info_document_loader("azure")
    from azure.storage.blob import BlobClient

    connection_string = get_required_env("AZURE_BLOB_STORAGE_CONNECTION_STRING")
    container_name = f"ragstack-ci-{uuid.uuid4()}"
    blob_name = "data.txt"

    container_client = ContainerClient.from_connection_string(
        conn_str=connection_string, container_name=container_name
    )
    try:
        container_client.create_container()

        blob_client = BlobClient.from_connection_string(
            conn_str=connection_string,
            container_name=container_name,
            blob_name=blob_name,
        )

        try:
            blob_client.upload_blob(io.BytesIO(b"test data"))
            loader = AzureBlobStorageContainerLoader(
                conn_str=connection_string, container=container_name
            )
            docs = loader.load()

            for doc in docs:
                assert doc.page_content == "test data"
                print("got..")
                print(doc.metadata)
                assert os.path.basename(doc.metadata["source"]) == "data.txt"
        finally:
            blob_client.delete_blob()
    finally:
        container_client.delete_container()


@pytest.mark.skipif(
    not is_astra, reason="Skipping test because astradb is not supported"
)
def test_astradb_loader() -> None:
    set_current_test_info_document_loader("astradb")

    handler = AstraDBVectorStoreHandler(VectorStoreImplementation.ASTRADB)
    handler.before_test()
    astra_ref = handler.astra_ref

    collection = handler.default_astra_client.create_collection(astra_ref.collection)

    collection.insert_many([{"foo": "bar", "baz": "qux"}] * 20)
    collection.insert_many(
        [{"foo": "bar2", "baz": "qux"}] * 4 + [{"foo": "bar", "baz": "qux"}] * 4
    )

    loader = AstraDBLoader(
        astra_ref.collection,
        token=astra_ref.token,
        api_endpoint=astra_ref.api_endpoint,
        nb_prefetched=1,
        projection={"foo": 1},
        find_options={"limit": 22},
        filter_criteria={"foo": "bar"},
        page_content_mapper=lambda r: "Payload: " + json.dumps(r),
    )
    docs = loader.load()

    assert len(docs) == 22
    ids = set()
    for doc in docs:
        assert doc.page_content.startswith("Payload: ")
        content = json.loads(doc.page_content[9:])
        assert content["foo"] == "bar"
        assert "baz" not in content
        assert content["_id"] not in ids
        ids.add(content["_id"])
        assert doc.metadata == {
            "namespace": "default_keyspace",
            "api_endpoint": astra_ref.api_endpoint,
            "collection": astra_ref.collection,
        }
