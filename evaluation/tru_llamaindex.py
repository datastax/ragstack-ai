import tru_shared
import uuid

from llama_index import get_response_synthesizer, VectorStoreIndex, StorageContext, ServiceContext
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine

from trulens_eval import TruLlama

collection_name = "llama_512"
app_prefix = "llama_512"

framework = tru_shared.Framework.LLAMA_INDEX

vstore = tru_shared.getAstraVectorStore(framework, collection_name)
chatModel = tru_shared.getAzureChatModel(framework, "gpt-35-turbo", "0613")
embeddings = tru_shared.getAzureEmbeddingsModel(framework)
datasets, golden_set = tru_shared.getTestData()

service_context = ServiceContext.from_defaults(
    llm=chatModel,
    embed_model=embeddings,
)

storage_context = StorageContext.from_defaults(
    vector_store=vstore,
)

index = VectorStoreIndex.from_vector_store(
    vector_store=vstore,
    service_context=service_context,
)

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=4,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer(
    service_context=service_context
)

# assemble pipeline
pipeline = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer
)

tru = tru_shared.initTru()

feedbacks = tru_shared.getFeedbackFunctions(pipeline, golden_set)

shortUuid = str(uuid.uuid4())[9:13]

for name in datasets:
    app = f"{app_prefix}_{shortUuid}_{name}"
    tru_recorder = TruLlama(
        pipeline,
        app_id=app,
        feedbacks=feedbacks,
        feedback_mode="deferred",
    )
    for query in datasets[name]:
        with tru_recorder as recording:
            pipeline.query(query)
