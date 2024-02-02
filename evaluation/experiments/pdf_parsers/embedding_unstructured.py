# docker run -p 8000:8000 -d --rm --name unstructured-api downloads.unstructured.io/unstructured-io/unstructured-api:latest --port 8000 --host 0.0.0.0
import tru_shared

from langchain_community.document_loaders.unstructured import UnstructuredAPIFileLoader
from langchain.text_splitter import TokenTextSplitter

import os, glob

os.environ["ASTRA_DB_ENDPOINT"] = os.environ.get("ASTRA_DB_ENDPOINT_PDF_SPLITS_2")
os.environ["ASTRA_DB_TOKEN"] = os.environ.get("ASTRA_DB_TOKEN_PDF_SPLITS_2")

framework = tru_shared.Framework.LANG_CHAIN

splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)

api_url = "http://localhost:8000/general/v0/general"

def import_docs(collection_name, mode="elements", strategy="fast"):
    vector_store = tru_shared.get_astra_vector_store(framework, collection_name)

    for file_path in glob.glob('data/*/source_files/*.pdf'):
        print(f"Loading {file_path} into {collection_name}")
        loader = UnstructuredAPIFileLoader(file_path, mode=mode, strategy=strategy, url=api_url)
        vector_store.add_documents(splitter.split_documents(loader.load()))


import_docs("UnstructuredFileLoader_single", "single")
import_docs("UnstructuredFileLoader_elements", "elements")
