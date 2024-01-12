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

from e2e_tests.langchain.rag_application import format_docs

import numpy as np

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

def create_chain(llm: BaseLanguageModel, retriever: BaseRetriever) -> Runnable:
    prompt = PromptTemplate.from_template(PROMPT)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser(),
    )
    return chain


def run_trulens_evaluation(vector_store: VectorStore, llm: BaseLanguageModel):
    tru = _initialize_tru()
    chain = create_chain(vector_store, llm)

    feedback_functions = _feedback_functions(chain, llm)
    tru_recorder = TruChain(
        chain,
        feedbacks=feedback_functions,
    )

    with tru_recorder as recording:
        chain("When was MyFakeProductForTesting released for the first time?")

    tru_record = recording.records[0]
    print(tru_record)
    # TODO: FRAZ - theoretically..test this now on a subset of llms.
