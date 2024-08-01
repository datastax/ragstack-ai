import bz2
import tempfile
from abc import ABC, abstractmethod
from os import makedirs, path
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import aiohttp
from tqdm.asyncio import tqdm


class QueryItem():
    query: str
    metadata: Dict[str, Any]

    def __init__(self, query:str, metadata: Dict[str, Any]):
        self.query = query
        self.metadata = metadata


class BaseDataset(ABC):

    root_storage_path: str
    name: str
    _subsets: List[str] = []
    _query_items: List[QueryItem]
    _golden_set: List[Dict[str, str]]

    def __init__(
        self, dataset_name: str, root_storage_path: str = "datasets"
    ):
        self.name = dataset_name
        self.root_storage_path = root_storage_path
        self._query_items = []
        self._golden_set = []

    def storage_path(self) -> str:
        """returns the path where dataset files should be stored"""
        return path.join(self.root_storage_path, self.sub_storage_path())

    def list_files_at_path(self, path: str) -> List[str]:
        """lists all files at a path (excluding dot files)"""
        return [
            f.name
            for f in Path(path).iterdir()
            if f.is_file() and not f.name.startswith(".")
        ]

    @property
    def subsets(self) -> List[str]:
        return self._subsets

    @subsets.setter
    def subsets(self, value: List[str]) -> None:
        self._subsets = value

    @abstractmethod
    def sub_storage_path(self) -> str:
        """the sub-path to store the dataset in"""

    @abstractmethod
    def download_dataset(self) -> None:
        """downloads a dataset locally"""

    @abstractmethod
    def get_source_file_paths(self) -> List[str]:
        """gets a list of source file paths for for a dataset"""

    @abstractmethod
    def _load_query_items_and_golden_set(self) -> None:
        """loads query_items and golden_set"""

    def get_query_items(self) -> List[QueryItem]:
        """gets a list of query items for a dataset"""
        if len(self._query_items) == 0:
            self._load_query_items_and_golden_set()
        return self._query_items

    def get_golden_set(self) -> List[Dict[str, str]]:
        """gets the set of ground_truth answers for a dataset"""
        if len(self._golden_set) == 0:
            self._load_query_items_and_golden_set()
        return self._golden_set

    async def _download_file(
        self, session: aiohttp.ClientSession, url: str, temp_file_path: str
    ) -> None:
        timeout = aiohttp.ClientTimeout(total=6000)
        async with session.get(url, timeout=timeout) as response:
            file_size = int(response.headers.get("Content-Length", 0))
            chunk_size = 1024
            with tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                desc=f'Downloading {url.split("/")[-1]}',
            ) as progress_bar:
                async with aiofiles.open(temp_file_path, "wb") as temp_file:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        await temp_file.write(chunk)
                        progress_bar.update(len(chunk))

    async def _decompress_file(
        self, temp_file_path: str, output_file_path: str
    ) -> None:
        makedirs(path.dirname(output_file_path), exist_ok=True)
        with open(temp_file_path, "rb") as temp_file:
            decompressed_size = 0
            with bz2.BZ2File(temp_file, "rb") as bz2_file:
                async with aiofiles.open(output_file_path, "wb") as output_file:
                    with tqdm(
                        unit="B",
                        unit_scale=True,
                        desc=f"Decompressing {output_file_path}",
                    ) as progress_bar:
                        while True:
                            chunk = bz2_file.read(1024)
                            if not chunk:
                                break
                            await output_file.write(chunk)
                            decompressed_size += len(chunk)
                            progress_bar.update(len(chunk))

    async def _download_and_decompress(
        self, url: str, output_file_path: str, force: bool
    ) -> None:
        if not force and path.exists(output_file_path):
            print(f"File {output_file_path} already exists. Skipping download.")
            return

        async with aiohttp.ClientSession() as session:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file_path = temp_file.name

                await self._download_file(session, url, temp_file_path)
                await self._decompress_file(temp_file_path, output_file_path)
