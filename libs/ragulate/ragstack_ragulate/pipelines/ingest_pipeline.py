import asyncio
from typing import List

from tqdm import tqdm

from ..logging_config import logger
from .base_pipeline import BasePipeline


class IngestPipeline(BasePipeline):

    @property
    def PIPELINE_TYPE(self):
        return "ingest"

    @property
    def get_reserved_params(self) -> List[str]:
        return ["file_path"]

    def ingest(self):

        logger.info(
            f"Starting ingest {self.recipe_name} on {self.script_path}/{self.method_name} with ingredients: {self.ingredients}  on datasets: {self.dataset_names()}"
        )

        source_files = []
        for dataset in self.datasets:
            source_files.extend(dataset.get_source_file_paths())

        source_files = list(set(source_files))

        ingest_method = self.get_method()
        for source_file in tqdm(source_files):
            if asyncio.iscoroutinefunction(ingest_method):
                asyncio.run(ingest_method(file_path=source_file, **self.ingredients))
            else:
                ingest_method(file_path=source_file, **self.ingredients)
