from typing import List, Optional

from ragstack_ragulate.analysis import Analysis


def setup_compare(subparsers):
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


def remove_sqlite_extension(s):
    """Remove the .sqlite extension from a string."""
    if s.endswith(".sqlite"):
        return s[:-7]
    return s


def call_compare(
    recipe: List[str],
    output: Optional[str] = "box-plots",
    **_,
):
    """Compare results from 2 (or more) recipes."""
    analysis = Analysis()

    recipes = [remove_sqlite_extension(r) for r in recipe]

    analysis.compare(recipes=recipes, output=output)
