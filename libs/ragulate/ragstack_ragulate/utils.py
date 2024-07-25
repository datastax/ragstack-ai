import re
from typing import Any, Dict, List, Union

from trulens_eval import Tru


def get_tru(recipe_name: str) -> Tru:
    """Return Tru for given recipe name."""
    Tru.RETRY_FAILED_SECONDS = 60
    Tru.RETRY_RUNNING_SECONDS = 30
    return Tru(
        database_url=f"sqlite:///{recipe_name}.sqlite", database_redact_keys=True
    )  # , name=name)


def convert_vars_to_ingredients(
    var_names: List[str], var_values: List[str]
) -> Dict[str, Any]:
    """Convert variables to ingredients."""
    params: Dict[str, Any] = {}
    for i, name in enumerate(var_names):
        params[name] = _convert_string(var_values[i])
    return params


def _convert_string(s: str) -> Union[str, int, float]:
    s = s.strip()
    if re.match(r"^\d+$", s):
        return int(s)
    if re.match(r"^\d*\.\d+$", s):
        return float(s)
    return s
