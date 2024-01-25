import tru_shared

from llama_index import get_response_synthesizer, VectorStoreIndex, StorageContext, ServiceContext
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine

framework = tru_shared.Framework.LLAMA_INDEX

collection_name = "llama_512"

vstore = tru_shared.get_astra_vector_store(framework, collection_name)
chatModel = tru_shared.get_azure_chat_model(framework, "gpt-35-turbo", "0613")
embeddings = tru_shared.get_azure_embeddings_model(framework)

service_context = ServiceContext.from_defaults(
    llm=chatModel, embed_model=embeddings)
storage_context = StorageContext.from_defaults(vector_store=vstore)
vector_store_index = VectorStoreIndex.from_vector_store(
    vector_store=vstore, service_context=service_context)

retriever = VectorIndexRetriever(index=vector_store_index, similarity_top_k=4)
response_synthesizer = get_response_synthesizer(
    service_context=service_context)

pipeline = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer
)

tru_shared.execute_experiment(framework, pipeline, "llama_512")
