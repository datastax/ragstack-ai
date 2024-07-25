from argparse import ArgumentParser, _SubParsersAction
from typing import Any, List

from ragstack_ragulate.analysis import Analysis

from .utils import remove_sqlite_extension


def setup_compare(subparsers: _SubParsersAction[ArgumentParser]) -> None:
    """Setup the compare command."""
    compare_parser = subparsers.add_parser(
        "compare", help="Compare results from 2 (or more) recipes"
    )
    compare_parser.add_argument(
        "-r",
        "--recipe",
        type=str,
        help="A recipe to compare. This can be passed multiple times.",
        required=True,
        action="append",
    )
    compare_parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The output method. Either box-plots (default) or histogram-grid",
        default="box-plots",
    )
    compare_parser.set_defaults(func=lambda args: call_compare(**vars(args)))


def call_compare(
    recipe: List[str],
    output: str = "box-plots",
    **_: Any,
) -> None:
    """Compare results from 2 (or more) recipes."""
    analysis = Analysis()

    recipes = [remove_sqlite_extension(r) for r in recipe]

    analysis.compare(recipes=recipes, output=output)
