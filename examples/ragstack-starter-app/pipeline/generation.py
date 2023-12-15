from typing import Optional
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate

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
    # TODO: Where is the retriever...
    chat_prompt = prompt + CONVERSATION_PROMPT_TEMPLATE
    chat_prompt_template = PromptTemplate(
        input_variables=["history", "input"], template=chat_prompt
    )
    conversation = ConversationChain(
        prompt=chat_prompt_template,
        llm=llm,
        verbose=True,
        memory=ConversationBufferMemory(ai_prefix="AI Assistant", return_messages=True),
    )
    return conversation


# TODO: Move this to models.py?
def open_ai_model(model: Optional[str] = None) -> ChatOpenAI:
    return ChatOpenAI(model=model) if model is not None else ChatOpenAI()
