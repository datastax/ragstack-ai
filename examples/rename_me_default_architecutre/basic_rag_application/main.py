import os
from dotenv import load_dotenv

from langchain.schema import Document

from ingestion import load_dataset
from chunking import token_text_split
from storage import open_ai_embeddings, initialize_astra_db, add_documents
from retrieval import as_retriever
from generation import basic_chat, basic_chat_with_memory, open_ai_model


def basic_astra_qa_model():
    load_dotenv()

    # Ingestion
    dataset = load_dataset("datastax/philosopher-quotes", split="train")
    documents = []
    for entry in dataset:
        metadata = {"author": entry["author"]}
        doc = Document(page_content=entry["quote"], metadata=metadata)
        documents.append(doc)

    # Chunking
    split_documents = token_text_split(documents, chunk_size=10, chunk_overlap=2)

    # Storage / Embedding
    collection = input("Collection: ")
    embedding = open_ai_embeddings()
    token = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
    api_endpoint = os.environ["ASTRA_DB_API_ENDPOINT"]

    vstore = initialize_astra_db(collection, embedding, token, api_endpoint)
    print(f"Adding {len(split_documents)} documents to AstraDB...")
    vstore.add_documents(split_documents)

    # Retrieval
    retriever = as_retriever(vstore)

    # Generation
    prompt = """
    You are a philosopher that draws inspiration from great thinkers of the past
    to craft well-thought answers to user questions. Use the provided context as the basis
    for your answers and do not make up new reasoning paths - just mix-and-match what you are given.
    Your answers must be concise and to the point, and refrain from answering about other topics than philosophy.
    """

    model = open_ai_model()

    print(f"Initializing model with prompt:\n{prompt}")
    chain = basic_chat(retriever, model, prompt)

    while True:
        query = input("Enter a question:\n")
        response = chain.invoke(query)
        print(f"Response:\n{response}")


def basic_astra_conversation_model():
    load_dotenv()

    # Ingestion
    dataset = load_dataset("datastax/philosopher-quotes", split="train")
    documents = []
    for entry in dataset:
        metadata = {"author": entry["author"]}
        doc = Document(page_content=entry["quote"], metadata=metadata)
        documents.append(doc)

    # Chunking
    split_documents = token_text_split(documents, chunk_size=10, chunk_overlap=2)

    # Storage / Embedding
    collection = input("Collection: ")
    embedding = open_ai_embeddings()
    token = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
    api_endpoint = os.environ["ASTRA_DB_API_ENDPOINT"]

    vstore = initialize_astra_db(collection, embedding, token, api_endpoint)
    print(f"Adding {len(split_documents)} documents to AstraDB...")
    vstore.add_documents(split_documents)

    # Retrieval
    retriever = as_retriever(vstore)

    # Generation
    prompt = """
    You are a philosopher that draws inspiration from great thinkers of the past
    to craft well-thought answers to user questions. Use the provided context as the basis
    for your answers and do not make up new reasoning paths - just mix-and-match what you are given.
    Your answers must be concise and to the point, and refrain from answering about other topics than philosophy.
    """

    model = open_ai_model()

    print(f"Initializing model with prompt:\n{prompt}")
    chain = basic_chat_with_memory(retriever, model, prompt)

    while True:
        query = input("Enter a question:\n")
        response = chain.invoke(query)
        print(f"Response:\n{response}")


basic_astra_conversation_model()
