import tru_shared

from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

import os

os.environ["ASTRA_DB_ENDPOINT"] = os.environ.get("ASTRA_DB_ENDPOINT_CHUNK_OVERLAP")
os.environ["ASTRA_DB_TOKEN"] = os.environ.get("ASTRA_DB_TOKEN_CHUNK_OVERLAP")

framework = tru_shared.Framework.LANG_CHAIN

chatModel = tru_shared.get_azure_chat_model(framework, "gpt-35-turbo", "0613")

prompt_template = """
Answer the question based only on the supplied context. If you don't know the answer, say: "I don't know".
Context: {context}
Question: {question}
Your answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

collection_name = "unstructured_single_0"
# collection_name = "unstructured_elements_0"
# collection_name = "unstructured_single_192"
# collection_name = "unstructured_elements_192"

vector_store = tru_shared.get_astra_vector_store(framework, collection_name)
pipeline = (
    {"context": vector_store.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | chatModel
    | StrOutputParser()
)

tru_shared.execute_experiment(framework, pipeline, collection_name)

print("Done!")
