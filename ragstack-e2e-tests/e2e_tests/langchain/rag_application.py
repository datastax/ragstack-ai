import logging
import time
from operator import itemgetter
from typing import Sequence, Callable

from langchain.evaluation import Criteria
from langchain.schema.vectorstore import VectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import (
    Runnable,
    RunnableLambda,
    RunnableMap,
)
from langchain.smith import RunEvalConfig
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.tracers import ConsoleCallbackHandler
from langchain import callbacks

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import (
    ConversationSummaryMemory,
)

from e2e_tests.test_utils.tracing import (
    record_langsmith_sharelink,
    ensure_langsmith_dataset,
    run_langchain_chain_on_dataset,
    get_langsmith_sharelink,
)

BASIC_QA_PROMPT = """
Answer the question based only on the supplied context. If you don't know the answer, say you don't know the answer.
Context: {context}
Question: {question}
Your answer:
"""

RESPONSE_TEMPLATE = """\
You are an expert programmer and problem-solver, tasked with answering any question \
about MyFakeProductForTesting.

Generate a comprehensive and informative answer of 80 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity.

You should use bullet points in your answer for readability. Put citations where they \
apply rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer.

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. 

<context>
    {context} 
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
"""

SAMPLE_DATA = [
    "MyFakeProductForTesting is a versatile testing tool designed to streamline the testing process for software developers, quality assurance professionals, and product testers. It provides a comprehensive solution for testing various aspects of applications and systems, ensuring robust performance and functionality.",
    # noqa: E501
    "MyFakeProductForTesting comes equipped with an advanced dynamic test scenario generator. This feature allows users to create realistic test scenarios by simulating various user interactions, system inputs, and environmental conditions. The dynamic nature of the generator ensures that tests are not only diverse but also adaptive to changes in the application under test.",
    # noqa: E501
    "The product includes an intelligent bug detection and analysis module. It not only identifies bugs and issues but also provides in-depth analysis and insights into the root causes. The system utilizes machine learning algorithms to categorize and prioritize bugs, making it easier for developers and testers to address critical issues first.",
    # noqa: E501
    "MyFakeProductForTesting first release happened in June 2020.",
]


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def create_chain(
    llm: BaseLanguageModel,
    retriever: BaseRetriever,
) -> Runnable:
    _context = RunnableMap(
        {
            "context": RunnableLambda(itemgetter("question")) | retriever | format_docs,
            "question": itemgetter("question"),
        }
    ).with_config(run_name="RetrieveDocs")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            ("human", "{question}"),
        ]
    )

    response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
        run_name="GenerateResponse",
    )
    return RunnableMap(
        {
            "answer": (
                {
                    "question": RunnableLambda(itemgetter("question")).with_config(
                        run_name="Itemgetter:question"
                    )
                }
                | _context
                | response_synthesizer
            )
        }
    )


def run_rag_custom_chain(
    vector_store: VectorStore, llm: BaseLanguageModel, record_property: Callable
) -> None:
    vector_store.add_texts(SAMPLE_DATA)
    retriever = vector_store.as_retriever()
    answer_chain = create_chain(
        llm,
        retriever,
    )

    ensure_langsmith_dataset(
        name="ragstack-ci-rag-custom-chain",
        input={
            "question": "When was released MyFakeProductForTesting for the first time ?"
        },
        output={"answer": "MyFakeProductForTesting was released in June 2020"},
    )

    runs = run_langchain_chain_on_dataset(
        dataset_name="ragstack-ci-rag-custom-chain",
        chain_factory=lambda: answer_chain,
        run_eval_config=RunEvalConfig(
            evaluators=[
                "context_qa",
                "cot_qa",
                RunEvalConfig.LabeledCriteria(Criteria.RELEVANCE),
                RunEvalConfig.LabeledCriteria(Criteria.HELPFULNESS),
                RunEvalConfig.LabeledCriteria(Criteria.COHERENCE),
            ]
        ),
    )
    if len(runs) != 1:
        raise ValueError(f"Expected 1 run, got {len(runs)}")
    actual_run = runs[0]
    logging.info(
        "Got response: " + str(actual_run.output) + " error: " + str(actual_run.error)
    )
    record_langsmith_sharelink(actual_run.run_id, record_property)

    for feedback in actual_run.feedbacks:
        logging.info(
            f"Feedback for {feedback.key} is {feedback.score} with value {feedback.value} for run {actual_run.run_id}"
        )
        xml_key = feedback.key.replace(" ", "_").lower()
        xml_value = f"{feedback.value} (score: {feedback.score})"
        record_property(f"langsmith_feedback_{xml_key}", xml_value)
        record_property(
            f"langsmith_feedback_{xml_key}_url",
            get_langsmith_sharelink(run_id=feedback.eval_run_id),
        )

    assert actual_run.error is None
    assert actual_run.output is not None
    assert "2020" in actual_run.output["answer"]


def run_conversational_rag(
    vector_store: VectorStore,
    llm: BaseLanguageModel,
    chat_memory: BaseChatMessageHistory,
    record_property,
) -> None:
    logging.info("Starting to add texts to vector store")
    start = time.perf_counter_ns()
    vector_store.add_texts(SAMPLE_DATA)
    logging.info(f"Added texts in {(time.perf_counter_ns() - start) / 1e9} seconds")
    retriever = vector_store.as_retriever()
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        chat_memory=chat_memory,
    )
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        verbose=True,
        memory=memory,
        callbacks=[ConsoleCallbackHandler()],
    )

    with callbacks.collect_runs() as cb:
        result = conversation.invoke({"question": "what is MyFakeProductForTesting?"})
        run_id = cb.traced_runs[0].id
        record_langsmith_sharelink(run_id, record_property)
        logging.info("First result: " + str(result))

    with callbacks.collect_runs() as cb:
        result = conversation.invoke({"question": "and when was it released?"})
        run_id = cb.traced_runs[0].id
        record_langsmith_sharelink(run_id, record_property)
        logging.info("Second result: " + str(result))

    assert "2020" in result["answer"]
