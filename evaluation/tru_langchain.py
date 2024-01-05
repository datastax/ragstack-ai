import tru_shared
import uuid

from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

from trulens_eval import TruChain

collection_name = "open_ai_512"
app_prefix = "lc_512"

framework = tru_shared.Framework.LANG_CHAIN

vstore = tru_shared.getAstraVectorStore(framework, collection_name)
chatModel = tru_shared.getAzureChatModel(framework, "gpt-35-turbo", "0613")
embeddings = tru_shared.getAzureEmbeddingsModel(framework)
datasets, golden_set = tru_shared.getTestData()


prompt_template = """
Answer the question based only on the supplied context. If you don't know the answer, say: "I don't know".
Context: {context}
Question: {question}
Your answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

pipeline = (
    {"context": vstore.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | chatModel
    | StrOutputParser()
)

tru = tru_shared.initTru()

feedbacks = tru_shared.getFeedbackFunctions(pipeline, golden_set)

shortUuid = str(uuid.uuid4())[9:13]

for name in datasets:
    app = f"{app_prefix}_{shortUuid}_{name}"
    tru_recorder = TruChain(
        pipeline,
        app_id=app,
        feedbacks=feedbacks,
        feedback_mode="deferred",
    )
    for query in datasets[name]:
        with tru_recorder as recording:
            pipeline.invoke(query)
