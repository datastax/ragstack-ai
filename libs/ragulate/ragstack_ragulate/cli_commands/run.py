from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragstack_ragulate.analysis import Analysis
from ragstack_ragulate.config import ConfigParser
from ragstack_ragulate.logging_config import logger
from ragstack_ragulate.pipelines import IngestPipeline, QueryPipeline

if TYPE_CHECKING:
    from argparse import ArgumentParser, _SubParsersAction


def setup_run(subparsers: _SubParsersAction[ArgumentParser]) -> None:
    """Setup the run command."""
    run_parser = subparsers.add_parser(
        "run", help="Run an experiment from a config file"
    )
    run_parser.add_argument(
        "config_file",
        type=str,
        help=(
            "The name of the yaml config_file that contains the recipes for your "
            "experiment."
        ),
    )
    run_parser.set_defaults(func=lambda args: call_run(**vars(args)))


def call_run(config_file: str, **_: Any) -> None:
    """Run an experiment from a config file."""
    config_parser = ConfigParser.from_file(file_path=config_file)
    config = config_parser.get_config()

    ingest_pipelines: list[IngestPipeline] = []
    query_pipelines: list[QueryPipeline] = []

    for dataset in config.datasets.values():
        dataset.download_dataset()

    for name, recipe in config.recipes.items():
        if recipe.ingest is not None:
            ingest_pipelines.append(
                IngestPipeline(
                    recipe_name=name,
                    script_path=recipe.ingest.script,
                    method_name=recipe.ingest.method,
                    ingredients=recipe.ingredients,
                    datasets=list(config.datasets.values()),
                )
            )
        if recipe.query is not None:
            query_pipelines.append(
                QueryPipeline(
                    recipe_name=name,
                    script_path=recipe.query.script,
                    method_name=recipe.query.method,
                    ingredients=recipe.ingredients,
                    datasets=list(config.datasets.values()),
                )
            )

    logger.debug("Found these ingest pipelines:")
    for ingest_pipeline in ingest_pipelines:
        logger.debug(f"\t{ingest_pipeline.key()}")

    ingest_pipelines = list(set(ingest_pipelines))

    logger.debug("Narrowed down to these ingest pipelines:")
    for ingest_pipeline in ingest_pipelines:
        logger.debug(f"\t{ingest_pipeline.key()}")

    logger.debug("Found these query pipelines:")
    for query_pipeline in query_pipelines:
        logger.debug(f"\t{query_pipeline.key()}")

    query_pipelines = list(set(query_pipelines))

    logger.debug("Narrowed down to these query pipelines:")
    for query_pipeline in query_pipelines:
        logger.debug(f"\t{query_pipeline.key()}")

    for ingest_pipeline in ingest_pipelines:
        ingest_pipeline.ingest()

    for query_pipeline in query_pipelines:
        query_pipeline.query()

    recipe_names = list(config.recipes.keys())

    analysis = Analysis()
    analysis.compare(recipes=recipe_names)
