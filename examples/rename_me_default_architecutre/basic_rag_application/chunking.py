from langchain.schema import Document
from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter


def token_text_split(documents, chunk_size, chunk_overlap) -> list[Document]:
    """
    Split a list of documents into chunks.

    Args:
        documents (List[Document]): A list of documents to be split.
        chunk_size (int): The size of each text chunk.
        overlap_size (int): The size of the overlap between consecutive chunks.

    Returns:
        List[Document]: A list of smaller text chunks as Document objects.
    """
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(texts)} chunks.")
    return texts


def recursive_character_text_split(
    documents, chunk_size, overlap_size
) -> list[Document]:
    """
    Split a list of documents into smaller character-based text chunks using recursion.

    Args:
        documents (list): A list of documents to be split.
        chunk_size (int): The size of each text chunk.
        overlap_size (int): The size of the overlap between consecutive chunks.

    Returns:
        list[Document]: A list of smaller text chunks as Document objects.
    """

    # Maybe want to use token text directly instead of character text
    # TODO: Test results of tokens
    text_splitter = RecursiveCharacterTextSplitter(chunk_size, overlap_size)
    texts = text_splitter.split_documents(documents)
    return texts
