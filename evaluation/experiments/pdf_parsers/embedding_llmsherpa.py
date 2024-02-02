# docker run -d -p 5010:5001 -v $(pwd)/data:/data ghcr.io/nlmatics/nlm-ingestor:latest

import tru_shared

from llmsherpa.readers import LayoutPDFReader
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader

import os, glob

os.environ["ASTRA_DB_ENDPOINT"] = os.environ.get("ASTRA_DB_ENDPOINT_PDF_SPLITS_2")
os.environ["ASTRA_DB_TOKEN"] = os.environ.get("ASTRA_DB_TOKEN_PDF_SPLITS_2")

framework = tru_shared.Framework.LANG_CHAIN

splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)

def import_docs(llmSherpa_api_url, collection_name):
    vector_store = tru_shared.get_astra_vector_store(framework, collection_name)

    pdf_reader = LayoutPDFReader(llmSherpa_api_url)

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


llmSherpa_api_url = "http://localhost:5010/api/parseDocument?renderFormat=all"

import_docs(llmSherpa_api_url, "LayoutPDFReader_base")
import_docs(llmSherpa_api_url + "&useNewIndentParser=yes", "LayoutPDFReader_new")
