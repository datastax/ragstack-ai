from langchain.document_loaders import BSHTMLLoader, DirectoryLoader, TextLoader, PDFMinerLoader, UnstructuredMarkdownLoader
from langchain.vectorstores.astradb import AstraDB
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter

from dotenv import load_dotenv
import os

load_dotenv()

data_loaders = {
    "html": {"loader": BSHTMLLoader, "kwargs": {}},
    "md": {"loader": UnstructuredMarkdownLoader, "kwargs": {}},
    "pdf": {"loader": PDFMinerLoader, "kwargs": {"concatenate_pages": True}},
    "txt": {"loader": TextLoader, "kwargs": {}},
}

docs = []

for extension in data_loaders:
    print(f"Loading {extension} files...")
    loader_cls = data_loaders[extension]["loader"]
    loader_kwargs = data_loaders[extension]["kwargs"]
    loader = DirectoryLoader(
        'data', glob=f"*/source_files/*.{extension}", show_progress=True, loader_cls=loader_cls, loader_kwargs=loader_kwargs)
    docs.extend(loader.load())

print(f"\nLoaded {len(docs)} documents.")

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2023-05-15"
)

# chunk_sizes = [128, 256, 512, 1024, 2048] # chunk size of 1024 is too big for astraPy vector store currently. max of 5000 bytes per entry.
chunk_sizes = [128, 256, 512, 768]

names = []
vstores = {}
splitters = {}

for size in chunk_sizes:
    name = f"open_ai_{size}"
    names.append(name)
    vstores[name] = AstraDB(collection_name=name, embedding=embeddings, token=os.getenv(
        "ASTRA_DB_TOKEN"), api_endpoint=os.getenv("ASTRA_DB_ENDPOINT"))
    splitters[name] = TokenTextSplitter(chunk_size=size, chunk_overlap=0)

print("ALL DONE! YAY!")
