import tru_shared

from llama_index.core import Settings, get_response_synthesizer, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding

import os

os.environ["ASTRA_DB_ENDPOINT"] = os.environ.get("ASTRA_DB_ENDPOINT_PDF_SPLITS")
os.environ["ASTRA_DB_TOKEN"] = os.environ.get("ASTRA_DB_TOKEN_PDF_SPLITS")

framework = tru_shared.Framework.LLAMA_INDEX

collection_name = "baseline"
# collection_name = "llm_sherpa"
# collection_name = "unstructured"
# collection_name = "llama_parse_text"
# collection_name = "llama_parse_markdown"

Settings.llm = tru_shared.get_azure_chat_model(framework, "gpt-35-turbo", "0613")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

response_synthesizer = get_response_synthesizer()

vector_store = tru_shared.get_astra_vector_store(framework, collection_name)
vector_store_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
retriever = VectorIndexRetriever(index=vector_store_index, similarity_top_k=5)

pipeline = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer
)

tru_shared.execute_experiment(framework, pipeline, collection_name)

print("done")
