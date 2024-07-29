from __future__ import annotations

import asyncio

from tqdm import tqdm
from typing_extensions import override

from ragstack_ragulate.logging_config import logger

from .base_pipeline import BasePipeline


class IngestPipeline(BasePipeline):
    """Ingest pipeline."""

    @property
    @override
    def pipeline_type(self) -> str:
        return "ingest"

    @property
    @override
    def get_reserved_params(self) -> list[str]:
        return ["file_path"]

    def ingest(self) -> None:
        """Run the ingest pipeline."""
        logger.info(
            f"Starting ingest {self.recipe_name} "
            f"on {self.script_path}/{self.method_name} "
            f"with ingredients: {self.ingredients} "
            f"on datasets: {self.dataset_names()}"
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
