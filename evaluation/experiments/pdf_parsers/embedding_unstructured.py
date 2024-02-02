# docker run -p 8000:8000 -d --rm --name unstructured-api downloads.unstructured.io/unstructured-io/unstructured-api:latest --port 8000 --host 0.0.0.0

from dotenv import load_dotenv
from langchain_community.document_loaders.unstructured import UnstructuredAPIFileLoader
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores.astradb import AstraDB
from langchain_openai import AzureOpenAIEmbeddings

import os
import glob

load_dotenv()

open_ai_embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2023-05-15"
)

splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)

api_url = "http://localhost:8000/general/v0/general"

def import_docs(collection_name, mode="elements", strategy="fast"):
    vector_store = AstraDB(
        collection_name=collection_name,
        embedding=open_ai_embeddings,
        api_endpoint=os.environ.get("ASTRA_DB_ENDPOINT_PDF_SPLITS_2"),
        token=os.environ.get("ASTRA_DB_TOKEN_PDF_SPLITS_2")
    )

    for file_path in glob.glob('data/*/source_files/*.pdf'):
        print(f"Loading {file_path} into {collection_name}")
        loader = UnstructuredAPIFileLoader(file_path, mode=mode, strategy=strategy, url=api_url)
        vector_store.add_documents(splitter.split_documents(loader.load()))


import_docs("UnstructuredFileLoader_single", "single")
import_docs("UnstructuredFileLoader_elements", "elements")
