from dotenv import load_dotenv
from langchain_community.document_loaders.pdf import PyPDFium2Loader, PyMuPDFLoader, PyPDFLoader, PDFMinerLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores.astradb import AstraDB
from langchain_openai import AzureOpenAIEmbeddings
import os

load_dotenv()

os.environ["ASTRA_DB_ENDPOINT"] = os.environ.get("ASTRA_DB_ENDPOINT_PDF_SPLITS")
os.environ["ASTRA_DB_TOKEN"] = os.environ.get("ASTRA_DB_TOKEN_PDF_SPLITS")

open_ai_embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2023-05-15"
)

collection_loader_map = {
    "PyPDFium2Loader" : { "loader": PyPDFium2Loader, "kwargs": {}},
    "PyMuPDFLoader" : { "loader": PyMuPDFLoader, "kwargs": {}},
    "PyPDFLoader" : { "loader": PyPDFLoader, "kwargs": {}},
    "PDFMinerLoader_by_page" : { "loader": PDFMinerLoader, "kwargs": {"concatenate_pages": False}},
    "PDFMinerLoader_by_pdf" : { "loader": PDFMinerLoader, "kwargs": {"concatenate_pages": True}},
}

for collection_name in collection_loader_map:
    vector_store = AstraDB(
        collection_name=collection_name,
        embedding=open_ai_embeddings,
        token=os.getenv("ASTRA_DB_TOKEN"),
        api_endpoint=os.getenv("ASTRA_DB_ENDPOINT")
    )

    print(f"Loading PDFs into {collection_name}:")
    loader_cls = collection_loader_map[collection_name]["loader"]
    loader_kwargs = collection_loader_map[collection_name]["kwargs"]
    loader = DirectoryLoader('data/', glob=f"*/source_files/*.pdf", show_progress=True, loader_cls=loader_cls, loader_kwargs=loader_kwargs)

    splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)
    vector_store.add_documents(splitter.split_documents(loader.load()))
