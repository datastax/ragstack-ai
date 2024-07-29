from __future__ import annotations

from typing import Any


def dict_to_string(d: dict[str, Any]) -> str:
    """Convert dictionary to string."""
    parts = []

    for key, value in d.items():
        parts.append(f"{key}_{value}")

    return "_".join(parts)
