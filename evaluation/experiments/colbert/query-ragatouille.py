from typing import List

import numpy as np
from dotenv import load_dotenv
from trulens_eval import Feedback, Tru, TruLlama
from trulens_eval.app import App
from trulens_eval.feedback import Groundedness, GroundTruthAgreement
from trulens_eval.feedback.provider import AzureOpenAI

load_dotenv()

import json
import os
import time
import uuid
from typing import List

from llama_index.core import QueryBundle, Settings, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever as LlamaBaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.llms.azure_openai import AzureOpenAI as LlamaAzureChatOpenAI
from ragatouille import RAGPretrainedModel

start_time = time.time()


azure_llama = LlamaAzureChatOpenAI(
    deployment_name="gpt-35-turbo",
    model="gpt-35-turbo",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-05-15",
    model_version="0613",
    temperature=0,
)

Settings.llm = azure_llama

print("Azure OpenAI is configured.")


def get_test_data():
    base_path = "./data"

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

# use a short uuid to ensure that multiple experiments with the same name don't collide in the DB
shortUuid = str(uuid.uuid4())[9:13]
datasets, golden_set = get_test_data()

print ("text data loaded.")

if os.getenv("TRULENS_DB_CONN_STRING"):
    tru = Tru(database_url=os.getenv("TRULENS_DB_CONN_STRING"))
else:
    tru = Tru()

def get_feedback_functions(pipeline, golden_set):
    # Initialize provider class
    azureOpenAI = AzureOpenAI(deployment_name="gpt-35-turbo-16k")

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


def get_recorder(pipeline, app_id: str, golden_set : [], feedback_mode : str = "deferred"):
    feedbacks = get_feedback_functions(pipeline, golden_set)
    return TruLlama(
        pipeline,
        app_id=app_id,
        feedbacks=feedbacks,
        feedback_mode=feedback_mode,
    )

RAG = RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/my_index")


class LlamaRagatouilleRetriever(LlamaBaseRetriever):
    """Custom retriever that uses Astra DB RAGatouille embeddings"""

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        results = RAG.search(query, k=5)
        nodes = []
        for result in results:
            body = result['content']
            del result['content']
            score = result['score']
            del result['score']

            node = TextNode(extra_info=result, text=body)
            nodes.append(NodeWithScore(node=node, score=score))
        return nodes


llama_ragatouille_retriever = LlamaRagatouilleRetriever()

# define response synthesizer
response_synthesizer = get_response_synthesizer()

# assemble query engine
colbert_query_engine = RetrieverQueryEngine(
    retriever=llama_ragatouille_retriever,
    response_synthesizer=response_synthesizer,
)

for dataset_name in datasets:
    app_id = f"RAGatouille#{shortUuid}#{dataset_name}"
    print(f"Starting processing App: {app_id}")
    tru_recorder = get_recorder(colbert_query_engine, app_id, golden_set)
    for query in datasets[dataset_name]:
        print(f"Starting query:{query}")
        try:
            with tru_recorder as recording:
                colbert_query_engine.query(query)
        except:
            print(f"Query: '{query}' caused exception, skipping.")

duration = time.time() - start_time
print(f"It took {duration} seconds to query the documents via RAGatouille.")
