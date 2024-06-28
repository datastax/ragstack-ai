import os

import inflection

from .base_dataset import BaseDataset
from .crag_dataset import CragDataset
from .llama_dataset import LlamaDataset


def find_dataset(name: str) -> BaseDataset:
    root_path = "datasets"
    if not os.path.exists(root_path):
        raise ValueError("please download a dataset before using ingest or query")

    name = inflection.underscore(name)
    for kind in os.listdir(root_path):
        kind_path = os.path.join(root_path, kind)
        if os.path.isdir(kind_path):
            for dataset in os.listdir(kind_path):
                dataset_path = os.path.join(kind_path, dataset)
                if os.path.isdir(dataset_path):
                    if dataset.lower() == name:
                        return get_dataset(name, kind)

    """ searches for a downloaded dataset with this name. if found, returns it."""
    return get_dataset(name, "llama")


def get_dataset(name: str, kind: str) -> BaseDataset:
    kind = kind.lower()
    if kind == "llama":
        return LlamaDataset(dataset_name=name)
    elif kind == "crag":
        return CragDataset(dataset_name=name)

    raise NotImplementedError("only llama and crag datasets are currently supported")
