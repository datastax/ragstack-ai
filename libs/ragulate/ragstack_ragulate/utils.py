import re
from typing import Any, Dict, List

from trulens_eval import Tru


def get_tru(recipe_name: str) -> Tru:
    Tru.RETRY_FAILED_SECONDS = 60
    Tru.RETRY_RUNNING_SECONDS = 30
    return Tru(
        database_url=f"sqlite:///{recipe_name}.sqlite", database_redact_keys=True
    )  # , name=name)


def convert_vars_to_ingredients(
    var_names: List[str], var_values: List[str]
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for i, name in enumerate(var_names):
        params[name] = convert_string(var_values[i])
    return params


def convert_string(s):
    s = s.strip()
    if re.match(r"^\d+$", s):
        return int(s)
    elif re.match(r"^\d*\.\d+$", s):
        return float(s)
    else:
        return s
