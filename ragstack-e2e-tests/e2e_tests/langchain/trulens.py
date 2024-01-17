from trulens_eval import TruChain, Feedback, Tru
from trulens_eval.feedback.provider import Langchain
from trulens_eval.feedback import Groundedness
from trulens_eval.app import App
from trulens_eval.schema import FeedbackResult

from langchain.schema.vectorstore import VectorStore
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever


import numpy as np
from concurrent.futures import as_completed

PROMPT = """
Answer the question based only on the supplied context. If you don't know the answer, say you don't know the answer.
Context: {context}
Question: {question}
Your answer:
"""


def _feedback_functions(chain: Runnable, llm: BaseLanguageModel) -> list[Feedback]:
    provider = Langchain(chain=llm)
    context = App.select_context(chain)

    grounded = Groundedness(groundedness_provider=provider)
    f_groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons)
        .on(context.collect())  # collect context chunks into a list
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )
    f_qa_relevance = Feedback(provider.relevance_with_cot_reasons).on_input_output()
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

    with tru_recorder as recording:
        chain.invoke("When was MyFakeProductForTesting released for the first time?")

    tru_record = recording.get()

    # Wait for the feedback results to complete
    for feedback_future in as_completed(tru_record.feedback_results):
        _, feedback_result = feedback_future.result()

        feedback_result: FeedbackResult

        # basic verification that feedback results were computed
        assert feedback_result.result > 0.0
