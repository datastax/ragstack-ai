from argparse import ArgumentParser, _SubParsersAction
from typing import Any

from ragstack_ragulate.dashboard import run_dashboard

from .utils import remove_sqlite_extension


def setup_dashboard(subparsers: _SubParsersAction[ArgumentParser]) -> None:
    """Setup the dashboard command."""
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Show the tru-lens dashboard for a recipe. Can be helpful for debugging.",
    )
    dashboard_parser.add_argument(
        "-r",
        "--recipe",
        type=str,
        help="A recipe to see the dashboard for.",
        required=True,
    )
    dashboard_parser.add_argument(
        "-p",
        "--port",
        type=int,
        help="Port to show the dashboard on, default 8501",
        default=8501,
    )
    dashboard_parser.set_defaults(func=lambda args: call_dashboard(**vars(args)))


def call_dashboard(
    recipe: str,
    port: int,
    **_: Any,
) -> None:
    """Runs the TruLens dashboard."""
    recipe_name = remove_sqlite_extension(recipe)
    run_dashboard(recipe_name=recipe_name, port=port)
