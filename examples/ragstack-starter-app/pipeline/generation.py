from typing import Optional
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate

CHAT_PROMPT_TEMPLATE = """
\n
CONTEXT: {context}

QUESTION: {question}

YOUR ANSWER:
"""

CONVERSATION_PROMPT_TEMPLATE = """
\n
Current conversation:
{history}

Context:
{context}

Human: 
{input}

AI Assistant:
"""


### TODO: So maybe this is more of "setup" of the retrieval?
### Perhaps generation is _just_ the invocation of the LLM given some retrieved context
### that is already been through post-processing, etc. (i.e. the retrieval pipeline)
### And generation is the final step that has the least amount of our code.

### generation:
### - models.py
### - prompt.py
### - outputs? (any post-processing of the LLM output?)
###
### retrieval:
### - chains (obviously, conversation chains are about providing context to the LLM, so retrieval).
###
### ingestion:
### - chunking.py
### - splitting.py
### - storage.py
###
### streamlit-app:
### - streamlit_app.py
### - resources/
###
### evaluation:
###


def basic_chat(retriever: VectorStoreRetriever, llm: BaseChatModel, prompt: str):
    chat_prompt = prompt + CHAT_PROMPT_TEMPLATE
    chat_prompt_template = ChatPromptTemplate.from_template(chat_prompt)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | chat_prompt_template
        | llm
        | StrOutputParser()
    )
    return chain


def basic_chat_with_memory(
    retriever: VectorStoreRetriever, llm: BaseChatModel, prompt: str
):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer",
    )

    qa_prompt = PromptTemplate(
        input_variables=["chat_history", "question", "context"], template=prompt
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        get_chat_history=lambda h: h,
        output_key="answer",
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        # verbose=True,
        # return_source_documents=True,
    )

    return chain


# TODO: Move this to models.py?
def open_ai_model(model: Optional[str] = None) -> ChatOpenAI:
    return ChatOpenAI(model=model) if model is not None else ChatOpenAI()
