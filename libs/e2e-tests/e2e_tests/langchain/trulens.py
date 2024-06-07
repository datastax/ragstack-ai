from trulens_eval import TruChain, Feedback, Tru
from trulens_eval.feedback.provider import Langchain
from trulens_eval.app import App
from trulens_eval.schema.feedback import FeedbackResult

from langchain.schema.vectorstore import VectorStore
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever

from e2e_tests.langchain.rag_application import (
    BASIC_QA_PROMPT,
    SAMPLE_DATA,
    format_docs,
)

import numpy as np
from concurrent.futures import as_completed


def _feedback_functions(chain: Runnable, llm: BaseLanguageModel) -> list[Feedback]:
    provider = Langchain(chain=llm)
    context = App.select_context(chain)

    f_groundedness = (
        Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
        .on(context.collect())  # collect context chunks into a list
        .on_output()
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
    # We can use the default db url and reset before each run.
    tru = Tru()
    tru.reset_database()
    return tru


def _create_chain(retriever: VectorStoreRetriever, llm: BaseLanguageModel) -> Runnable:
    prompt = PromptTemplate.from_template(BASIC_QA_PROMPT)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def run_trulens_evaluation(vector_store: VectorStore, llm: BaseLanguageModel):
    """
    Executes the TruLens evaluation process.
    """
    vector_store.add_texts(SAMPLE_DATA)
    _initialize_tru()
    retriever = vector_store.as_retriever()
    chain = _create_chain(retriever=retriever, llm=llm)

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
    for feedback_def, feedback_future in tru_record.feedback_and_future_results:
        feedback_result = feedback_future.result()
        # basic verification that feedback results were computed
        assert feedback_result.result is not None
