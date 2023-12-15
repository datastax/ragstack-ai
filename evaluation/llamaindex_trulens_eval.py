# run with: nohup python3 llama_512.py > llama_512.log 2>&1 &

collection_name = "llama_512"

from llama_index.vector_stores import AstraDBVectorStore
from llama_index.llms import AzureOpenAI
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index import  VectorStoreIndex, StorageContext, ServiceContext
from llama_index import get_response_synthesizer
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor

from trulens_eval import Tru

from dotenv import load_dotenv
import os, json, time

load_dotenv()

## init astraDB vector store

astra_db_store = AstraDBVectorStore(
    collection_name=collection_name,
    api_endpoint=os.getenv("ASTRA_DB_ENDPOINT"),
    token=os.getenv("ASTRA_DB_TOKEN"),
    embedding_dimension=1536,
)

## setup azure LLMs

temperature = 0.0

gpt_35_turbo = AzureOpenAI(
    deployment_name="gpt-35-turbo",
    model="gpt-35-turbo",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-05-15",
    model_version="0613",
    temperature=temperature,
)

gpt_35_turbo_16k = AzureOpenAI(
    deployment_name="gpt-35-turbo-16k",
    model="gpt-35-turbo-16k",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-05-15",
    model_version="0613",
    temperature=temperature,
)

gpt_4 = AzureOpenAI(
    deployment_name="gpt-4",
    model="gpt-4",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-05-15",
    model_version="1106-preview",
    temperature=temperature,
)

gpt_4_32k = AzureOpenAI(
    deployment_name="gpt-4-32k",
    model="gpt-4-32k",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-05-15",
    model_version="0613",
    temperature=temperature,
)

embed_model = AzureOpenAIEmbedding(
    deployment_name="text-embedding-ada-002",
    model="text-embedding-ada-002",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-05-15",
    temperature=temperature,
)

## setup query engine

service_context = ServiceContext.from_defaults(
    llm=gpt_35_turbo,
    embed_model=embed_model,
)

storage_context = StorageContext.from_defaults(
    vector_store=astra_db_store,
)

index = VectorStoreIndex.from_vector_store(
    vector_store=astra_db_store,
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

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    # node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)

## Setup TruLens

tru = Tru(database_url=os.getenv("TRULENS_DB_CONN_STRING"))

## Load Datasets

base_path = "./data/"

datasets = {}
golden_set = []

for name in os.listdir(base_path):
    if os.path.isdir(os.path.join(base_path, name)):
        datasets[name] = []
        with open(os.path.join(base_path, name, "rag_dataset.json")) as f:
            examples = json.load(f)['examples']
            for e in examples:
                datasets[name].append(e["query"])
                golden_set.append({
                    "query": e["query"],
                    "response": e["reference_answer"],
                })
            print("Loaded dataset: ", name)

## expose Tru AzureOpenAI class

from trulens_eval.feedback.provider import OpenAI
from trulens_eval.feedback.provider.endpoint.openai import OpenAIClient
import openai as oai

# note that this is a name change from AzureOpenAI to not collide with the llamaindex AzureOpenAI model
class TruAzureOpenAI(OpenAI):
    def __init__(self, endpoint=None, deployment_name="gpt-35-turbo", **kwargs):
        kwargs["client"] = OpenAIClient(client=oai.AzureOpenAI(**kwargs))
        super().__init__(
            endpoint=endpoint, model_engine=deployment_name, **kwargs
        )  # need to include pydantic.BaseModel.__init__

    def _create_chat_completion(self, *args, **kwargs):
        return super()._create_chat_completion(*args, **kwargs)

## init feedback functions

# this class isn't exposed yet :(, so use copy above for now :)
# from trulens_eval.feedback.provider import AzureOpenAI as TruAzureOpenAI
from trulens_eval.feedback import Groundedness, GroundTruthAgreement
from trulens_eval import Select, TruLlama, Feedback
import numpy as np
# Initialize provider class
truAzureOpenAI = TruAzureOpenAI(deployment_name="gpt-35-turbo")

context_selection = TruLlama.select_source_nodes().node.text

grounded = Groundedness(groundedness_provider=truAzureOpenAI)
# Define a groundedness feedback function
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons)
    .on(context_selection)
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

# Question/answer relevance between overall question and answer.
f_answer_relevance = Feedback(truAzureOpenAI.relevance_with_cot_reasons).on_input_output()
# Question/statement relevance between question and each context chunk.
f_context_relevance = (
    Feedback(truAzureOpenAI.qs_relevance_with_cot_reasons)
    .on_input()
    .on(context_selection)
    .aggregate(np.mean)
)

# GroundTruth for comparing the Answer to the Ground-Truth Answer
ground_truth_collection = GroundTruthAgreement(golden_set, provider=truAzureOpenAI)
f_answer_correctness = (
    Feedback(ground_truth_collection.agreement_measure)
    .on_input_output()
)

# run evaluation

for name in datasets:
    app = f"{name}_{collection_name}"
    print(f"Starting work on {app}")
    tru_recorder = TruLlama(
        query_engine,
        app_id=app,
        feedbacks=[f_answer_relevance, f_context_relevance, f_groundedness, f_answer_correctness]
    )
    for query in datasets[name]:
        time.sleep(10)
        with tru_recorder as recording:
            query_engine.query(query)

print("ALL DONE! YAY!")
