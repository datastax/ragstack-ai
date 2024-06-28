from typing import List

from ragstack_ragulate.datasets import find_dataset
from ragstack_ragulate.pipelines import IngestPipeline

from ..utils import convert_vars_to_ingredients


def setup_ingest(subparsers):
    ingest_parser = subparsers.add_parser("ingest", help="Run an ingest pipeline")
    ingest_parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="A unique name for the ingest pipeline",
        required=True,
    )
    ingest_parser.add_argument(
        "-s",
        "--script_path",
        type=str,
        help="The path to the python script that contains the ingest method",
        required=True,
    )
    ingest_parser.add_argument(
        "-m",
        "--method-name",
        type=str,
        help="The name of the method in the script to run ingest",
        required=True,
    )
    ingest_parser.add_argument(
        "--var-name",
        type=str,
        help="The name of a variable in the ingest script. This should be paired with a `--var-value` argument and can be passed multiple times.",
        action="append",
    )
    ingest_parser.add_argument(
        "--var-value",
        type=str,
        help="The value of a variable in the ingest script. This should be paired with a `--var-name` argument and can be passed multiple times.",
        action="append",
    )
    ingest_parser.add_argument(
        "--dataset",
        type=str,
        help="The name of a dataset to ingest. This can be passed multiple times.",
        action="append",
    )
    ingest_parser.set_defaults(func=lambda args: call_ingest(**vars(args)))

    def call_ingest(
        name: str,
        script_path: str,
        method_name: str,
        var_name: List[str],
        var_value: List[str],
        dataset: List[str],
        **kwargs,
    ):

        datasets = [find_dataset(name=name) for name in dataset]

        ingredients = convert_vars_to_ingredients(
            var_names=var_name, var_values=var_value
        )

        ingest_pipeline = IngestPipeline(
            recipe_name=name,
            script_path=script_path,
            method_name=method_name,
            ingredients=ingredients,
            datasets=datasets,
        )
        ingest_pipeline.ingest()
