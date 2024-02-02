# docker run -d -p 5010:5001 -v $(pwd)/data:/data ghcr.io/nlmatics/nlm-ingestor:latest

from dotenv import load_dotenv
from llmsherpa.readers import LayoutPDFReader
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores.astradb import AstraDB
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_openai import AzureOpenAIEmbeddings

import os
import glob

load_dotenv()

open_ai_embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2023-05-15"
)

splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)


def import_docs(llmsherpa_api_url, collection_name):
    vector_store = AstraDB(
        collection_name=collection_name,
        embedding=open_ai_embeddings,
        api_endpoint=os.environ.get("ASTRA_DB_ENDPOINT_PDF_SPLITS_2"),
        token=os.environ.get("ASTRA_DB_TOKEN_PDF_SPLITS_2")
    )

    pdf_reader = LayoutPDFReader(llmsherpa_api_url)

    for file_path in glob.glob('data/*/source_files/*.pdf'):
        print(f"Loading {file_path} into {collection_name}")
        try:
            pdf = pdf_reader.read_pdf(file_path)

            docs = []
            for chunk in pdf.chunks():
                docs.append(Document(page_content=chunk.to_context_text()))

            vector_store.add_documents(splitter.split_documents(docs))
        except Exception as e:
            print(f"Exception loading file with LayoutPDFReader, using PyPDFLoader instead")
            vector_store.add_documents(splitter.split_documents(PyPDFLoader(file_path).load()))


llmsherpa_api_url = "http://localhost:5010/api/parseDocument?renderFormat=all"

import_docs(llmsherpa_api_url, "LayoutPDFReader_base")
import_docs(llmsherpa_api_url + "&useNewIndentParser=yes",
            "LayoutPDFReader_new")
