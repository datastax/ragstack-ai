import json, os, time
import numpy as np

from datetime import datetime
from dotenv import load_dotenv
from enum import Enum

from trulens_eval import Tru, Feedback
from trulens_eval.app import App
from trulens_eval.feedback.provider import AzureOpenAI
from trulens_eval.feedback import Groundedness, GroundTruthAgreement

from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index.llms import AzureOpenAI as LlamaAzureChatOpenAI
from llama_index.vector_stores import AstraDBVectorStore

from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
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


def getTestData():
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

def initTru():
    return Tru(database_url=os.getenv("TRULENS_DB_CONN_STRING"))


def getFeedbackFunctions(pipeline, golden_set):
    # Initialize provider class
    azureOpenAI = AzureOpenAI(deployment_name="gpt-35-turbo")

    context = App.select_context(pipeline)

    # Define a groundedness feedback function
    grounded = Groundedness(groundedness_provider=azureOpenAI)
    f_groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons)
        .on(context.collect()).on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )

    # Question/answer relevance between overall question and answer.
    f_answer_relevance = (
        Feedback(azureOpenAI.relevance_with_cot_reasons)
        .on_input_output()
    )

    # Question/statement relevance between question and each context chunk.
    f_context_relevance = (
        Feedback(azureOpenAI.qs_relevance_with_cot_reasons)
        .on_input().on(context)
        .aggregate(np.mean)
    )

    # GroundTruth for comparing the Answer to the Ground-Truth Answer
    ground_truth_collection = GroundTruthAgreement(golden_set, provider=azureOpenAI)
    f_answer_correctness = (
        Feedback(ground_truth_collection.agreement_measure)
        .on_input_output()
    )
    return [f_answer_relevance, f_context_relevance, f_groundedness, f_answer_correctness]


def waitForResults(tru, app, index):
    # it normally takes about 10 seconds to get results
    # so delay until that time, and then check more frequently
    print(f"waiting for results on app: {app} index: {index}")
    start = datetime.now()
    time.sleep(7)
    while True:
        time.sleep(2)
        df, feedbackColumns = tru.get_records_and_feedback([app])
        row = df.loc[index]
        completeCount = 0
        for fbCol in feedbackColumns:
            if not np.isnan(row[fbCol]):
                completeCount += 1
        if completeCount == len(feedbackColumns):
            return
        else:
            print(f"index: {index} has completeCount: {completeCount}, continuing to wait")
        if (datetime.now() - start).total_seconds() > 30:
            print("timeout, giving up")
            return


def getAzureChatModel(framework: Framework, deployment_name: str, model_version: str):
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
        raise Exception(f"Unknown framework: {framework} specified for getChatModel()")


def getAzureEmbeddingsModel(framework: Framework):
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
        raise Exception(f"Unknown framework: {framework} specified for getEmbeddingsModel()")


def getAstraVectorStore(framework: Framework, collection_name: str):
    if framework == Framework.LANG_CHAIN:
        return AstraDB(
            collection_name=collection_name,
            embedding=getAzureEmbeddingsModel(framework),
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
        raise Exception(f"Unknown framework: {framework} specified for getAstraVectorStore()")

