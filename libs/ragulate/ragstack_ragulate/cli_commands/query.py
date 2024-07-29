from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragstack_ragulate.datasets import find_dataset
from ragstack_ragulate.pipelines import QueryPipeline
from ragstack_ragulate.utils import convert_vars_to_ingredients

if TYPE_CHECKING:
    from argparse import ArgumentParser, _SubParsersAction


def setup_query(subparsers: _SubParsersAction[ArgumentParser]) -> None:
    """Setup the query command."""
    query_parser = subparsers.add_parser("query", help="Run a query pipeline")
    query_parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="A unique name for the query pipeline",
        required=True,
    )
    query_parser.add_argument(
        "-s",
        "--script",
        type=str,
        help="The path to the python script that contains the query method",
        required=True,
    )
    query_parser.add_argument(
        "-m",
        "--method",
        type=str,
        help="The name of the method in the script to run query",
        required=True,
    )
    query_parser.add_argument(
        "--var-name",
        type=str,
        help="The name of a variable in the query script. This should be paired with a "
        "`--var-value` argument and can be passed multiple times.",
        action="append",
    )
    query_parser.add_argument(
        "--var-value",
        type=str,
        help="The value of a variable in the query script. This should be paired with "
        "a `--var-name` argument and can be passed multiple times.",
        action="append",
    )
    query_parser.add_argument(
        "--dataset",
        type=str,
        help="The name of a dataset to query. This can be passed multiple times.",
        action="append",
    )
    query_parser.add_argument(
        "--subset",
        type=str,
        help="The subset of the dataset to query. "
        "Only valid when a single dataset is passed.",
        action="append",
    )
    query_parser.add_argument(
        "--sample",
        type=float,
        help="A decimal percentage of the queries to sample for the test. "
        "Default is 1.0.",
        default=1.0,
    )
    query_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed to use for query sampling. Ensures reproducibility of tests.",
    )
    query_parser.add_argument(
        "--restart",
        help="Flag to restart the query process instead of resuming. "
        "WARNING: this will delete all existing data for this query name, "
        "not just the data for the tagged datasets.",
        action="store_true",
    )
    query_parser.add_argument(
        "--provider",
        type=str,
        help="The name of the LLM Provider to use for Evaluation.",
        choices=[
            "OpenAI",
            "AzureOpenAI",
            "Bedrock",
            "LiteLLM",
            "Langchain",
            "Huggingface",
        ],
        default="OpenAI",
    )
    query_parser.add_argument(
        "--model",
        type=str,
        help="The name or id of the LLM model or deployment to use for Evaluation. "
        "Generally used in combination with the `--provider` param.",
    )
    query_parser.set_defaults(func=lambda args: call_query(**vars(args)))

    def call_query(
        name: str,
        script: str,
        method: str,
        var_name: list[str],
        var_value: list[str],
        dataset: list[str],
        subset: list[str],
        sample: float,
        seed: int,
        restart: bool,
        provider: str,
        model: str,
        **_: Any,
    ) -> None:
        """Run a query pipeline."""
        if sample <= 0.0 or sample > 1.0:
            raise ValueError("Sample percent must be between 0 and 1")

        datasets = [find_dataset(name=name) for name in dataset]

        if subset is not None and len(subset) > 0:
            if len(datasets) > 1:
                raise ValueError(
                    "Only can set `subset` param when there is one dataset"
                )
            datasets[0].subsets = subset

        ingredients = convert_vars_to_ingredients(
            var_names=var_name, var_values=var_value
        )

        query_pipeline = QueryPipeline(
            recipe_name=name,
            script_path=script,
            method_name=method,
            ingredients=ingredients,
            datasets=datasets,
            sample_percent=sample,
            random_seed=seed,
            restart_pipeline=restart,
            llm_provider=provider,
            model_name=model,
        )
        query_pipeline.query()
