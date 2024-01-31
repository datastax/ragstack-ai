import tru_shared

from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import glob, os

os.environ["ASTRA_DB_ENDPOINT"] = os.environ.get("ASTRA_DB_ENDPOINT_PDF_SPLITS")
os.environ["ASTRA_DB_TOKEN"] = os.environ.get("ASTRA_DB_TOKEN_PDF_SPLITS")

framework = tru_shared.Framework.LANG_CHAIN

chatModel = tru_shared.get_azure_chat_model(framework, "gpt-35-turbo", "0613")
embeddings = tru_shared.get_azure_embeddings_model(framework)

pdf_datasets = []
for file_path in glob.glob('data/*/source_files/*.pdf'):
    dataset = file_path.split("/")[1]
    if dataset not in pdf_datasets:
        pdf_datasets.append(dataset)

collection_names = ["PyPDFium2Loader", "PyMuPDFLoader", "PyPDFLoader", "PDFMinerLoader_by_page", "PDFMinerLoader_by_pdf"]

prompt_template = """
Answer the question based only on the supplied context. If you don't know the answer, say: "I don't know".
Context: {context}
Question: {question}
Your answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

for collection_name in collection_names:
    vstore = tru_shared.get_astra_vector_store(framework, collection_name)
    pipeline = (
        {"context": vstore.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | chatModel
        | StrOutputParser()
    )

    tru_shared.execute_experiment(framework, pipeline, collection_name, pdf_datasets)

