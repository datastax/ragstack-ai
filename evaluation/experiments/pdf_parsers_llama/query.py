import tru_shared

from llama_index import get_response_synthesizer, VectorStoreIndex, ServiceContext
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine

import os

os.environ["ASTRA_DB_ENDPOINT"] = os.environ.get("ASTRA_DB_ENDPOINT_PDF_SPLITS")
os.environ["ASTRA_DB_TOKEN"] = os.environ.get("ASTRA_DB_TOKEN_PDF_SPLITS")

framework = tru_shared.Framework.LLAMA_INDEX

# collection_name = "baseline"
collection_name = "llm_sherpa"
# collection_name = "unstructured"
# collection_name = "llama_parse_text"
# collection_name = "llama_parse_markdown"

llm = tru_shared.get_azure_chat_model(framework, "gpt-35-turbo", "0613")
embeddings = tru_shared.get_azure_embeddings_model(framework)
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embeddings)
response_synthesizer = get_response_synthesizer(service_context=service_context)

vector_store = tru_shared.get_astra_vector_store(framework, collection_name)
vector_store_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)
retriever = VectorIndexRetriever(index=vector_store_index, similarity_top_k=4)

pipeline = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer
)

tru_shared.execute_experiment(framework, pipeline, collection_name)

print("done")
