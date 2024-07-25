from argparse import ArgumentParser, _SubParsersAction
from typing import Any

from ragstack_ragulate.datasets import get_dataset


def setup_download(subparsers: _SubParsersAction[ArgumentParser]) -> None:
    """Setup the download command."""
    download_parser = subparsers.add_parser("download", help="Download a dataset")
    download_parser.add_argument(
        "dataset_name",
        type=str,
        help=(
            "The name of the dataset you want to download, "
            "such as `PaulGrahamEssayDataset`."
        ),
    )
    download_parser.add_argument(
        "-k",
        "--kind",
        type=str,
        help="The kind of dataset to download. Currently only `llama` is supported",
        required=True,
    )
    download_parser.set_defaults(func=lambda args: call_download(**vars(args)))


def call_download(dataset_name: str, kind: str, **_: Any) -> None:
    """Download a dataset."""
    dataset = get_dataset(name=dataset_name, kind=kind)
    dataset.download_dataset()
