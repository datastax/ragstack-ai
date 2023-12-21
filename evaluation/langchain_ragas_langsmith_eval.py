# run with: nohup python3 768.py > open_ai_768.log 2>&1 &

collection_name = "open_ai_256"

from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.smith import RunEvalConfig, run_on_dataset

from ragas.langchain.evalchain import RagasEvaluatorChain
from ragas.llms import LangchainLLM
from ragas.metrics import answer_correctness, answer_relevancy, context_recall, context_relevancy, answer_similarity, faithfulness

from typing import Any, Dict

import json
from dotenv import load_dotenv

load_dotenv()

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
from langchain.vectorstores.astradb import AstraDB
import os
vstore = AstraDB(
    collection_name=collection_name,
    embedding=embeddings,
    token=os.getenv("ASTRA_DB_TOKEN"),
    api_endpoint=os.getenv("ASTRA_DB_ENDPOINT")
)

## Setup LangSmith
from langsmith import Client
client = Client()


## Create/Load the datasets
base_path = "./data/"

datasets = {}

for name in os.listdir(base_path):
    if os.path.isdir(os.path.join(base_path, name)):
        if not client.has_dataset(dataset_name=name):
            # if not create a new one with the generated query examples
            dataset = client.create_dataset(
                dataset_name=name, description=f"{name} dataset"
            )

            with open(os.path.join(base_path, name, "rag_dataset.json")) as f:
                examples = json.load(f)['examples']

                for e in examples:
                    exp = client.create_example(
                        inputs={"query": e["query"]},
                        outputs={"ground_truths": [e["reference_answer"]]},
                        dataset_id=dataset.id,
                    )

                print("Created a new dataset: ", dataset.name)

        # load the (new) dataset
        datasets[name] = client.read_dataset(dataset_name=name)
        print("Loaded dataset: ", name)

# build a prompt
prompt_template = """
Answer the question based only on the supplied context. If you don't know the answer, say: "I don't know".
Context: {context}
Question: {question}
Your answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# factory function that return a new qa chain
# this is so state is not reused when running each example
def create_qa_chain(return_context=True):
    qa_chain = RetrievalQA.from_llm(
        llm=gpt_35_turbo,
        prompt=prompt,
        retriever=vstore.as_retriever(),
        return_source_documents=return_context,
    )
    return qa_chain

## force ragas evaluators to use azure models instead of openai models

# embeddings can be used as it is
answer_relevancy.embeddings = embeddings
answer_similarity.embeddings = embeddings

# wrappers around azure models
ragas_gpt4 = LangchainLLM(gpt_4)
ragas_gpt35 = LangchainLLM(gpt_35_turbo)

# patch the ragas evaluators
answer_correctness.llm = ragas_gpt35
answer_relevancy.llm = ragas_gpt35
answer_similarity.llm = ragas_gpt35
context_relevancy.llm = ragas_gpt35
context_recall.llm = ragas_gpt35
faithfulness.llm = ragas_gpt35

answer_correctness.answer_similarity = answer_similarity
answer_correctness.faithfulness = faithfulness

# wrap evaluators for LangSmith

faithfulness_chain = RagasEvaluatorChain(metric=faithfulness)
answer_cor_chain = RagasEvaluatorChain(metric=answer_correctness)
answer_rel_chain = RagasEvaluatorChain(metric=answer_relevancy)
context_rel_chain = RagasEvaluatorChain(metric=context_relevancy)
context_rec_chain = RagasEvaluatorChain(metric=context_recall)

# create a method to run evaluation on a dataset
def run_evaluation(dataset_name, project_metadata: Dict[str, Any], project_name: str) -> Dict[str, Any]:
    evaluation_config = RunEvalConfig(
        custom_evaluators=[
            context_rec_chain,
            answer_cor_chain,
            faithfulness_chain,
            answer_rel_chain,
            context_rel_chain,
        ],
        prediction_key="result",
    )

    return run_on_dataset(
        client,
        dataset_name,
        create_qa_chain,
        evaluation=evaluation_config,
        project_metadata=project_metadata,
        project_name=project_name,
        input_mapper=lambda x: x
    )

for name in datasets:
    for i in range(5):
        metadata = {"collection": collection_name, "eval-model": "gpt3.5-turbo", "dataset": name, "run": i}
        project_name = f"col:{collection_name} eval:gpt-3.5-turbo dataset:{name} run:{i}"
        print(f"Starting: {metadata}")
        result = run_evaluation(dataset_name=name, project_metadata=metadata, project_name=project_name)

print("ALL DONE! YAY!")
