import os

from typing import Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    BSHTMLLoader,
    DirectoryLoader,
    TextLoader,
    PDFMinerLoader,
    UnstructuredMarkdownLoader,
)
import requests

data_loaders = {
    "html": {"loader": BSHTMLLoader, "kwargs": {}},
    "md": {"loader": UnstructuredMarkdownLoader, "kwargs": {}},
    "pdf": {"loader": PDFMinerLoader, "kwargs": {"concatenate_pages": True}},
    "txt": {"loader": TextLoader, "kwargs": {}},
}


def load_from_dir(path: str) -> list[Document]:
    """
    Load a dataset from a directory.

    Args:
        path (str): The path to the directory containing the dataset.

    Returns:
        list[Document]: A list of documents representing the loaded dataset.
    """

    documents = []
    for extension in data_loaders:
        print(f"Loading {extension} files...")
        loader_cls = data_loaders[extension]["loader"]
        loader_kwargs = data_loaders[extension]["kwargs"]
        loader = DirectoryLoader(
            path,
            glob=f"*.{extension}",
            show_progress=True,
            loader_cls=loader_cls,
            loader_kwargs=loader_kwargs,
        )
        documents.extend(loader.load())
        print(f"Loaded {len(documents)} {extension} files.")
        documents


def load_dataset(path: str, split: Optional[str] = None) -> list[Document]:
    from datasets import load_dataset

    """
    Load a dataset using HuggingFace's `load_dataset`

    Args:
        path (str): The path to the directory containing the dataset.
        split (Optional[str]): The name of the dataset split to load (e.g., 'train', 'test', 'validation').
            If not provided, the default behavior is to load the entire dataset.

    Returns:
        list[Document]: A list of documents representing the loaded dataset.
    """
    print(f"Loading dataset {path}...")
    dataset = load_dataset(path, split=split)
    print("Loaded dataset")
    return dataset


def load_from_file(path: str) -> list[Document]:
    _root, extension = os.path.splitext(path)
    if data_loaders.get(extension[1:]):
        loader = data_loaders[extension[1:]]["loader"]
        documents = loader(path).load()
        print(f"Loaded {len(documents)} {extension} files.")
        return documents
    else:
        raise ValueError(f"Could not load dataset with extension {extension}")


def load_from_url(url: str, local_dest: str) -> list[Document]:
    """
    Load a dataset from a URL and put it at `local_dest`.

    Args:
        path (str): The URL to the dataset.
        local_dest (str): The destination for the downloaded file.

    Returns:
        list[Document]: A list of documents representing the loaded dataset.
    """
    if os.path.exists(local_dest):
        print(f"Dataset already exists at {local_dest}. Loading...")
    else:
        print(f"Downloading dataset from {url}...")
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Could not download dataset from {url}")

        # Create the directory if it doesn't exist
        dir = os.path.dirname(local_dest)
        os.makedirs(dir, exist_ok=True)

        with open(local_dest, "wb") as f:
            f.write(response.content)

        print(f"Downloaded dataset to {local_dest}")

    return load_from_file(local_dest)
