# run with: nohup python3 768.py > open_ai_768.log 2>&1 &

collection_name = "open_ai_512"

from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores.astradb import AstraDB
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

from trulens_eval import Tru
from trulens_eval.feedback.provider import OpenAI
from trulens_eval.feedback.provider.endpoint.openai import OpenAIClient

from dotenv import load_dotenv

import openai as oai
import os
import time
import json


load_dotenv()

## init some things

temperature = 0.0

gpt_35_turbo = AzureChatOpenAI(
    azure_deployment="gpt-35-turbo",
    openai_api_version="2023-05-15",
    model_version="0613",
    temperature=temperature,
)

gpt_35_turbo_16k = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_deployment="gpt-35-turbo-16k",
    model_version="0613",
    temperature=temperature,
)

gpt_4 = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_deployment="gpt-4",
    model_version="1106-preview",
    temperature=temperature,
)

gpt_4_32k = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_deployment="gpt-4-32k",
    model_version="0613",
    temperature=temperature,
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2023-05-15"
)

## Setup Vector Store

vstore = AstraDB(
    collection_name=collection_name,
    embedding=embeddings,
    token=os.getenv("ASTRA_DB_TOKEN"),
    api_endpoint=os.getenv("ASTRA_DB_ENDPOINT")
)

## Setup TruLens

tru = Tru(database_url=os.getenv("TRULENS_DB_CONN_STRING"))

## expose AzureOpenAI class
class AzureOpenAI(OpenAI):
    """Out of the box feedback functions calling AzureOpenAI APIs.
    Has the same functionality as OpenAI out of the box feedback functions.
    """

    def __init__(self, endpoint=None, deployment_name="gpt-35-turbo", **kwargs):
        # NOTE(piotrm): pydantic adds endpoint to the signature of this
        # constructor if we don't include it explicitly, even though we set it
        # down below. Adding it as None here as a temporary hack.
        """
        Wrapper to use Azure OpenAI. Please export the following env variables

        - OPENAI_API_BASE
        - OPENAI_API_VERSION
        - OPENAI_API_KEY

        **Usage:**
        ```python
        from trulens_eval.feedback.provider.openai import OpenAI
        openai_provider = AzureOpenAI(deployment_id="...")
        ```

        Args:
            deployment_name (str, required): The name of the deployment.
                Defaults to "gpt-35-turbo".
            endpoint (Endpoint): Internal Usage for DB serialization
        """

        kwargs["client"] = OpenAIClient(client=oai.AzureOpenAI(**kwargs))
        super().__init__(
            endpoint=endpoint, model_engine=deployment_name, **kwargs
        )  # need to include pydantic.BaseModel.__init__

    def _create_chat_completion(self, *args, **kwargs):
        """
        We need to pass `engine`
        """
        return super()._create_chat_completion(*args, **kwargs)

# load datasets
base_path = "./data/"

datasets = {}
golden_set = []

for name in os.listdir(base_path):
    if os.path.isdir(os.path.join(base_path, name)):
        datasets[name] = []
        with open(os.path.join(base_path, name, "rag_dataset.json")) as f:
            examples = json.load(f)['examples']
            index = 0
            for e in examples:
                datasets[name].append(e["query"])
                golden_set.append({
                    "query": e["query"],
                    "response": e["reference_answer"],
                })
                index += 1
            print("Loaded dataset: ", name)

## initialize feedback functions

# this class isn't exposed yet :(, so use copy above for now :)
# from trulens_eval.feedback.provider import AzureOpenAI
from trulens_eval.feedback import Groundedness, GroundTruthAgreement
from trulens_eval import Select, TruChain, Feedback
from trulens_eval.schema import FeedbackResult
import numpy as np
# Initialize provider class
azureOpenAI = AzureOpenAI(deployment_name="gpt-35-turbo")

grounded = Groundedness(groundedness_provider=azureOpenAI)
# Define a groundedness feedback function
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons)
    .on(Select.RecordCalls.first.invoke.rets.context)
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

# Question/answer relevance between overall question and answer.
f_answer_relevance = Feedback(azureOpenAI.relevance_with_cot_reasons).on_input_output()
# Question/statement relevance between question and each context chunk.
f_context_relevance = (
    Feedback(azureOpenAI.qs_relevance_with_cot_reasons)
    .on(Select.RecordCalls.first.invoke.args.input)
    .on(Select.RecordCalls.first.invoke.rets.context)
    .aggregate(np.mean)
)

# GroundTruth for comparing the Answer to the Ground-Truth Answer
ground_truth_collection = GroundTruthAgreement(golden_set, provider=azureOpenAI)
f_answer_correctness = (
    Feedback(ground_truth_collection.agreement_measure)
    .on_input_output()
)

# build a prompt and chain

prompt_template = """
Answer the question based only on the supplied context. If you don't know the answer, say: "I don't know".
Context: {context}
Question: {question}
Your answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

rag_chain = (
    {"context": vstore.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | gpt_35_turbo
    | StrOutputParser()
)

# run evaluation

for name in datasets:
    project = f"{name}_{collection_name}"
    print(f"Starting work on {project}")
    tru_recorder = TruChain(
        rag_chain,
        app_id=project,
        feedbacks=[f_answer_relevance, f_context_relevance, f_groundedness, f_answer_correctness],
        feedback_mode="deferred",
    )
    for query in datasets[name]:
        time.sleep(5)
        with tru_recorder as recording:
            rag_chain.invoke(query)

print("ALL DONE! YAY!")
