from typing import Any, Dict

from pydantic import BaseModel

from ragstack_ragulate.datasets import BaseDataset


class Step(BaseModel):
    name: str
    script: str
    method: str


class Recipe(BaseModel):
    name: str
    ingest: Step | None
    query: Step
    cleanup: Step | None
    ingredients: Dict[str, Any]


class Config(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    recipes: Dict[str, Recipe] = {}
    datasets: Dict[str, BaseDataset] = {}
