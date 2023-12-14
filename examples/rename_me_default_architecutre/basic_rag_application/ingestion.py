from typing import Optional
from langchain.schema import Document
from langchain.document_loaders import (
    BSHTMLLoader,
    DirectoryLoader,
    TextLoader,
    PDFMinerLoader,
    UnstructuredMarkdownLoader,
)


def load_data(path: str) -> list[Document]:
    data_loaders = {
        "html": {"loader": BSHTMLLoader, "kwargs": {}},
        "md": {"loader": UnstructuredMarkdownLoader, "kwargs": {}},
        "pdf": {"loader": PDFMinerLoader, "kwargs": {"concatenate_pages": True}},
        "txt": {"loader": TextLoader, "kwargs": {}},
    }
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
    print(f"Loaded dataset")
    return dataset
