import json
from os import path
from typing import Dict, List, Optional, Tuple

import inflection
from llama_index.core.llama_dataset import download
from llama_index.core.llama_dataset.download import (
    LLAMA_DATASETS_LFS_URL,
    LLAMA_DATASETS_SOURCE_FILES_GITHUB_TREE_URL,
)

from ..logging_config import logger
from .base_dataset import BaseDataset


class LlamaDataset(BaseDataset):

    _llama_datasets_lfs_url: str
    _llama_datasets_source_files_tree_url: str

    def __init__(
        self, dataset_name: str, root_storage_path: Optional[str] = "datasets"
    ):
        super().__init__(dataset_name=dataset_name, root_storage_path=root_storage_path)
        self._llama_datasets_lfs_url: str = LLAMA_DATASETS_LFS_URL
        self._llama_datasets_source_files_tree_url: str = (
            LLAMA_DATASETS_SOURCE_FILES_GITHUB_TREE_URL
        )

    def sub_storage_path(self) -> str:
        return "llama"

    def _get_dataset_path(self) -> str:
        folder = inflection.underscore(self.name)
        folder = folder.removesuffix("_dataset")
        return path.join(self.storage_path(), folder)

    def download_dataset(self) -> None:
        """downloads a dataset locally"""
        download_dir = self._get_dataset_path()

        def download_by_name(name):
            download.download_llama_dataset(
                llama_dataset_class=name,
                download_dir=download_dir,
                llama_datasets_lfs_url=self._llama_datasets_lfs_url,
                llama_datasets_source_files_tree_url=self._llama_datasets_source_files_tree_url,
                show_progress=True,
                load_documents=False,
            )

        # to conform with naming scheme at LlamaHub
        name = self.name
        try:
            download_by_name(name=name)
        except:
            if not name.endswith("Dataset"):
                try:
                    download_by_name(name + "Dataset")
                except:
                    raise ValueError(f"Could not find {name} datset.")
            else:
                raise ValueError(f"Could not find {name} datset.")

        logger.info(f"Successfully downloaded {self.name} to {download_dir}")

    def get_source_file_paths(self) -> List[str]:
        """gets a list of source file paths for for a dataset"""
        source_path = path.join(self._get_dataset_path(), "source_files")
        return self.list_files_at_path(path=source_path)

    def get_queries_and_golden_set(self) -> Tuple[List[str], List[Dict[str, str]]]:
        """gets a list of queries and golden_truth answers for a dataset"""
        json_path = path.join(self._get_dataset_path(), "rag_dataset.json")
        with open(json_path, "r") as f:
            examples = json.load(f)["examples"]
            queries = [e["query"] for e in examples]
            golden_set = [
                {
                    "query": e["query"],
                    "response": e["reference_answer"],
                }
                for e in examples
            ]
            return queries, golden_set
