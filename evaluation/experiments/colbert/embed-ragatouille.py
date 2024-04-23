from dotenv import load_dotenv

load_dotenv()

import time

from llama_index.core import SimpleDirectoryReader
from ragatouille import RAGPretrainedModel

start_time = time.time()

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

reader = SimpleDirectoryReader(
    "./data/",
    recursive=True,
    required_exts=[".pdf", ".md", ".txt", ".html"]
)
docs = reader.load_data()

print(f"Loaded {len(docs)} documents")

doc_texts = [doc.text for doc in docs]

index_path = RAG.index(index_name="my_index", collection=doc_texts)

print(f"RAGatouille index exists at: {index_path}")

duration = time.time() - start_time
print(f"It took {duration} seconds to load the documents via RAGatouille.")