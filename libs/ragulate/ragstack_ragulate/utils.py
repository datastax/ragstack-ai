from __future__ import annotations

import re
from typing import Any

from trulens_eval import Tru


def get_tru(recipe_name: str) -> Tru:
    """Return Tru for given recipe name."""
    Tru.RETRY_FAILED_SECONDS = 60
    Tru.RETRY_RUNNING_SECONDS = 30
    return Tru(
        database_url=f"sqlite:///{recipe_name}.sqlite", database_redact_keys=True
    )  # , name=name)


def convert_vars_to_ingredients(
    var_names: list[str], var_values: list[str]
) -> dict[str, Any]:
    """Convert variables to ingredients."""
    params: dict[str, Any] = {}
    for i, name in enumerate(var_names):
        params[name] = _convert_string(var_values[i])
    return params


def _convert_string(s: str) -> str | int | float:
    s = s.strip()
    if re.match(r"^\d+$", s):
        return int(s)
    if re.match(r"^\d*\.\d+$", s):
        return float(s)
    return s
