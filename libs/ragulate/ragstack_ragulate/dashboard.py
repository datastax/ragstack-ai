from typing import Optional

from .utils import get_tru


def run_dashboard(recipe_name: str, port: Optional[int] = 8501) -> None:
    """Runs the TruLens dashboard."""
    tru = get_tru(recipe_name=recipe_name)
    tru.run_dashboard(port=port)
