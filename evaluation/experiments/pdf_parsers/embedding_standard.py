import tru_shared

from langchain_community.document_loaders.pdf import PyPDFium2Loader, PyMuPDFLoader, PyPDFLoader, PDFMinerLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import TokenTextSplitter

import os

os.environ["ASTRA_DB_ENDPOINT"] = os.environ.get("ASTRA_DB_ENDPOINT_PDF_SPLITS")
os.environ["ASTRA_DB_TOKEN"] = os.environ.get("ASTRA_DB_TOKEN_PDF_SPLITS")

framework = tru_shared.Framework.LANG_CHAIN

splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)

collection_loader_map = {
    "PyPDFium2Loader" : { "loader": PyPDFium2Loader, "kwargs": {}},
    "PyMuPDFLoader" : { "loader": PyMuPDFLoader, "kwargs": {}},
    "PyPDFLoader" : { "loader": PyPDFLoader, "kwargs": {}},
    "PDFMinerLoader_by_page" : { "loader": PDFMinerLoader, "kwargs": {"concatenate_pages": False}},
    "PDFMinerLoader_by_pdf" : { "loader": PDFMinerLoader, "kwargs": {"concatenate_pages": True}},
}

for collection_name in collection_loader_map:
    vector_store = tru_shared.get_astra_vector_store(framework, collection_name)

    print(f"Loading PDFs into {collection_name}:")
    loader_cls = collection_loader_map[collection_name]["loader"]
    loader_kwargs = collection_loader_map[collection_name]["kwargs"]
    loader = DirectoryLoader('data/', glob=f"*/source_files/*.pdf", show_progress=True, loader_cls=loader_cls, loader_kwargs=loader_kwargs)

    vector_store.add_documents(splitter.split_documents(loader.load()))
