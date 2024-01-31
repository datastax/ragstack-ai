import tru_shared

from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

framework = tru_shared.Framework.LANG_CHAIN

chatModel = tru_shared.get_azure_chat_model(framework, "gpt-35-turbo", "0613")
embeddings = tru_shared.get_azure_embeddings_model(framework)

prompt_template = """
Answer the question based only on the supplied context. If you don't know the answer, say: "I don't know".
Context: {context}
Question: {question}
Your answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

chunk_sizes = [128, 256, 512]
for size in chunk_sizes:
    vstore = tru_shared.get_astra_vector_store(framework, f"open_ai_{size}")
    pipeline = (
        {"context": vstore.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | chatModel
        | StrOutputParser()
    )

    tru_shared.execute_experiment(framework, pipeline, f"chunk_size_{size}")

