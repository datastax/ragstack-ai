from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ragstack_ragulate.datasets import BaseDataset  # noqa: TCH001


class Step(BaseModel):
    """Step of a recipe."""

    name: str
    script: str
    method: str


class Recipe(BaseModel):
    """Recipe object."""

    name: str
    ingest: Step | None
    query: Step
    cleanup: Step | None
    ingredients: dict[str, Any]


class Config(BaseModel):
    """Config object."""

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    recipes: dict[str, Recipe] = {}
    datasets: dict[str, BaseDataset] = {}
