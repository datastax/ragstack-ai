from typing import Optional
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.astradb import AstraDB
from langchain_core.embeddings import Embeddings
from langchain.vectorstores import VectorStore


def open_ai_embeddings(model: Optional[str] = None) -> OpenAIEmbeddings:
    """
    Load OpenAI embeddings.
    """

    return OpenAIEmbeddings(model=model) if model is not None else OpenAIEmbeddings()


def initialize_astra_db(
    collection: str, embedding: Embeddings, token: str, api_endpoint: str
) -> AstraDB:
    """
    Initialize an AstraDB instance with the specified parameters.

    Args:
        collection (str): The name of the collection in AstraDB.
        embedding (Embeddings): The embedding model.
        token (str): The authentication token for AstraDB.
        api_endpoint (str): The API endpoint URL for AstraDB.

    Returns:
        AstraDB: An instance of the AstraDB class.
    """
    vstore = AstraDB(
        collection_name=collection,
        embedding=embedding,
        token=token,
        api_endpoint=api_endpoint,
    )

    print("Astra vector store configured")
    return vstore


def add_documents(documents: list[Document], vstore: VectorStore):
    """
    Add documents to the AstraDB instance.

    Args:
        documents (list[Document]): A list of documents to be added.
        vstore (VectorStore): a vector store
    """
    vstore.add_documents(documents)
