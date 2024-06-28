from typing import Any, Dict


def dict_to_string(d: Dict[str, Any]) -> str:
    parts = []

    for key, value in d.items():
        parts.append(f"{key}_{value}")

    return "_".join(parts)
