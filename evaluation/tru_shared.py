import json
import os
import uuid
import numpy as np

from dotenv import load_dotenv
from enum import Enum

from trulens_eval import Tru, Feedback, TruChain, TruLlama
from trulens_eval.app import App
from trulens_eval.feedback.provider import AzureOpenAI
from trulens_eval.feedback import Groundedness, GroundTruthAgreement

from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index.llms import AzureOpenAI as LlamaAzureChatOpenAI
from llama_index.vector_stores import AstraDBVectorStore

from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores.astradb import AstraDB

# this code assumes the following env vars exist in a .env file:
#
# ASTRA_DB_ENDPOINT
# ASTRA_DB_TOKEN
# AZURE_OPENAI_ENDPOINT
# AZURE_OPENAI_API_KEY
# OPENAI_API_VERSION
# TRULENS_DB_CONN_STRING

load_dotenv()

temperature = 0

class Framework(Enum):
    LANG_CHAIN = "langChain"
    LLAMA_INDEX = "llamaIndex"


def get_test_data():
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
    return datasets, golden_set


def init_tru():
    if os.getenv("TRULENS_DB_CONN_STRING"):
        return Tru(database_url=os.getenv("TRULENS_DB_CONN_STRING"))
    else:
        return Tru()


def get_feedback_functions(pipeline, golden_set):
    # Initialize provider class
    azureOpenAI = AzureOpenAI(deployment_name="gpt-35-turbo")

    context = App.select_context(pipeline)

    # Define a groundedness feedback function
    grounded = Groundedness(groundedness_provider=azureOpenAI)
    f_groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons,
                 name="groundedness")
        .on(context.collect()).on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )

    # Question/answer relevance between overall question and answer.
    f_answer_relevance = (
        Feedback(azureOpenAI.relevance_with_cot_reasons,
                 name="answer_relevance")
        .on_input_output()
    )

    # Question/statement relevance between question and each context chunk.
    f_context_relevance = (
        Feedback(azureOpenAI.qs_relevance_with_cot_reasons,
                 name="context_relevance")
        .on_input().on(context)
        .aggregate(np.mean)
    )

    # GroundTruth for comparing the Answer to the Ground-Truth Answer
    ground_truth_collection = GroundTruthAgreement(
        golden_set, provider=azureOpenAI)
    f_answer_correctness = (
        Feedback(ground_truth_collection.agreement_measure,
                 name="answer_correctness")
        .on_input_output()
    )
    return [f_answer_relevance, f_context_relevance, f_groundedness, f_answer_correctness]


def get_recorder(framework: Framework, pipeline, app_id: str, golden_set: [], feedback_mode: str = "deferred"):
    feedbacks = get_feedback_functions(pipeline, golden_set)
    if framework == Framework.LANG_CHAIN:
        return TruChain(
            pipeline,
            app_id=app_id,
            feedbacks=feedbacks,
            feedback_mode=feedback_mode,
        )
    elif framework == Framework.LLAMA_INDEX:
        return TruLlama(
            pipeline,
            app_id=app_id,
            feedbacks=feedbacks,
            feedback_mode=feedback_mode,
        )
    else:
        raise Exception(
            f"Unknown framework: {framework} specified for get_recorder()")


def get_azure_chat_model(framework: Framework, deployment_name: str, model_version: str):
    if framework == Framework.LANG_CHAIN:
        return AzureChatOpenAI(
            azure_deployment=deployment_name,
            openai_api_version="2023-05-15",
            model_version=model_version,
            temperature=temperature,
        )
    elif framework == Framework.LLAMA_INDEX:
        return LlamaAzureChatOpenAI(
            deployment_name=deployment_name,
            model=deployment_name,
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2023-05-15",
            model_version=model_version,
            temperature=temperature,
        )
    else:
        raise Exception(
            f"Unknown framework: {framework} specified for getChatModel()")


def get_azure_embeddings_model(framework: Framework):
    if framework == Framework.LANG_CHAIN:
        return AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-ada-002",
            openai_api_version="2023-05-15"
        )
    elif framework == Framework.LLAMA_INDEX:
        return AzureOpenAIEmbedding(
            deployment_name="text-embedding-ada-002",
            model="text-embedding-ada-002",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2023-05-15",
            temperature=temperature,
        )
    else:
        raise Exception(
            f"Unknown framework: {framework} specified for getEmbeddingsModel()")


def get_astra_vector_store(framework: Framework, collection_name: str):
    if framework == Framework.LANG_CHAIN:
        return AstraDB(
            collection_name=collection_name,
            embedding=get_azure_embeddings_model(framework),
            token=os.getenv("ASTRA_DB_TOKEN"),
            api_endpoint=os.getenv("ASTRA_DB_ENDPOINT")
        )
    elif framework == Framework.LLAMA_INDEX:
        return AstraDBVectorStore(
            collection_name=collection_name,
            api_endpoint=os.getenv("ASTRA_DB_ENDPOINT"),
            token=os.getenv("ASTRA_DB_TOKEN"),
            embedding_dimension=1536,
        )
    else:
        raise Exception(
            f"Unknown framework: {framework} specified for get_astra_vector_store()")


def execute_query(framework: Framework, pipeline, query):
    if framework == Framework.LANG_CHAIN:
        pipeline.invoke(query)
    elif framework == Framework.LLAMA_INDEX:
        pipeline.query(query)
    else:
        raise Exception(
            f"Unknown framework: {framework} specified for execute_query()")


# runs the pipeline across all queries in all known datasets
def execute_experiment(framework: Framework, pipeline, experiment_name: str):
    tru = init_tru()

    # use a short uuid to ensure that multiple experiments with the same name don't collide in the DB
    shortUuid = str(uuid.uuid4())[9:13]
    datasets, golden_set = get_test_data()

    for dataset_name in datasets:
        app_id = f"{experiment_name}#{shortUuid}#{dataset_name}"
        tru_recorder = get_recorder(framework, pipeline, app_id, golden_set)
        for query in datasets[dataset_name]:
            try:
                with tru_recorder as recording:
                    execute_query(framework, pipeline, query)
            except:
                print(f"Query: '{query}' caused exception, skipping.")
