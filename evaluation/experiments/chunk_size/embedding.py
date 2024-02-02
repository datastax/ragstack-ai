import tru_shared

from langchain.document_loaders import BSHTMLLoader, DirectoryLoader, TextLoader
from langchain_community.document_loaders import PDFMinerLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import TokenTextSplitter

import os

os.environ["ASTRA_DB_ENDPOINT"] = os.environ.get("ASTRA_DB_ENDPOINT_CHUNK_SIZE")
os.environ["ASTRA_DB_TOKEN"] = os.environ.get("ASTRA_DB_TOKEN_CHUNK_SIZE")

framework = tru_shared.Framework.LANG_CHAIN

data_loaders = {
    "html": {"loader": BSHTMLLoader, "kwargs": {}},
    "md": {"loader": UnstructuredMarkdownLoader, "kwargs": {}},
    "pdf": {"loader": PDFMinerLoader, "kwargs": {"concatenate_pages": True}},
    "txt": {"loader": TextLoader, "kwargs": {}},
}

chunk_sizes = [128, 256, 512, 1024]
for chunk_size in chunk_sizes:
    collection_name = f"chunk_size_{chunk_size}"
    vector_store = tru_shared.get_astra_vector_store(framework, collection_name)
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=0)

    for extension in data_loaders:
        print(f"Loading {extension} files using chunk size {chunk_size}...")
        loader_cls = data_loaders[extension]["loader"]
        loader_kwargs = data_loaders[extension]["kwargs"]
        loader = DirectoryLoader(
            'data', glob=f"*/source_files/*.{extension}", loader_cls=loader_cls, loader_kwargs=loader_kwargs)
        vector_store.add_documents(splitter.split_documents(loader.load()))
