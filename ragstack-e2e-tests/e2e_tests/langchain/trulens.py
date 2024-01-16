from trulens_eval import TruChain, Feedback, Tru, Select
from trulens_eval.feedback.provider import Langchain
from trulens_eval.feedback import Groundedness, GroundTruthAgreement
from trulens_eval.app import App

from langchain.schema.vectorstore import VectorStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import Runnable
from pydantic import BaseModel

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever

from e2e_tests.langchain.rag_application import format_docs

import numpy as np

PROMPT = """
Answer the question based only on the supplied context. If you don't know the answer, say you don't know the answer.
Context: {context}
Question: {question}
Your answer:
"""


def _feedback_functions(chain: Runnable, llm: BaseLanguageModel) -> list[Feedback]:
    provider = Langchain(chain=llm)
    context = App.select_context(chain)

    # Groundedness is how well the answer is supported by the context.
    grounded = Groundedness(groundedness_provider=provider)
    f_groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons)
        .on(context.collect())  # collect context chunks into a list
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )

    # QA relevance is how relevant the answer is to the question.
    f_qa_relevance = Feedback(provider.relevance_with_cot_reasons).on_input_output()

    # Ground truth is how well the answer matches the ground truth from the dataset.
    # f_ground_truth = Feedback(
    #     GroundTruthAgreement(
    #         ground_truth=golden_set, provider=openai
    #     ).agreement_measure,
    #     name="Ground Truth",
    # ).on_input_output()

    # Context relevance is how relevant the retrieved context is to the question.
    f_context_relevance = (
        Feedback(provider.qs_relevance_with_cot_reasons)
        .on_input()
        .on(context.collect())
        .aggregate(np.mean)
    )
    return [f_groundedness, f_qa_relevance, f_context_relevance]


def _initialize_tru() -> Tru:
    # We can use the default db url, then ensure it's reset before each run.
    tru = Tru()
    tru.reset_database()
    return tru

    # "OpenAI",
    # "AzureOpenAI",
    # "Huggingface",
    # "LiteLLM",
    # "Bedrock",
    # "Langchain",


def create_chain(retriever: VectorStoreRetriever, llm: BaseLanguageModel) -> Runnable:
    prompt = PromptTemplate.from_template(PROMPT)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def run_trulens_evaluation(vector_store: VectorStore, llm: BaseLanguageModel):
    vector_store.add_texts(
        [
            "MyFakeProductForTesting is a versatile testing tool designed to streamline the testing process for software developers, quality assurance professionals, and product testers. It provides a comprehensive solution for testing various aspects of applications and systems, ensuring robust performance and functionality.",  # noqa: E501
            "MyFakeProductForTesting comes equipped with an advanced dynamic test scenario generator. This feature allows users to create realistic test scenarios by simulating various user interactions, system inputs, and environmental conditions. The dynamic nature of the generator ensures that tests are not only diverse but also adaptive to changes in the application under test.",  # noqa: E501
            "The product includes an intelligent bug detection and analysis module. It not only identifies bugs and issues but also provides in-depth analysis and insights into the root causes. The system utilizes machine learning algorithms to categorize and prioritize bugs, making it easier for developers and testers to address critical issues first.",  # noqa: E501
            "MyFakeProductForTesting first release happened in June 2020.",
        ]
    )
    _initialize_tru()
    retriever = vector_store.as_retriever()
    chain = create_chain(retriever=retriever, llm=llm)

    feedback_functions = _feedback_functions(chain=chain, llm=llm)
    tru_recorder = TruChain(
        chain,
        app_id="test",
        feedbacks=feedback_functions,
    )
    result = chain.invoke(
        "when was MyFakeProductForTesting released for the first time?"
    )
    print(result)

    with tru_recorder as recording:
        chain.invoke("When was MyFakeProductForTesting released for the first time?")

    tru_record = recording.records[0]
    print(tru_record)
