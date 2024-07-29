from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

    from ragstack_ragulate.datasets import BaseDataset


def load_module(file_path: str, name: str) -> ModuleType:
    """Load a module from a file path dynamically."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None:
        raise ValueError(f"Could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ValueError(f"No Module loader found for {file_path}")
    spec.loader.exec_module(module)
    return module


def get_method(script_path: str, pipeline_type: str, method_name: str) -> Any:
    """Return the method from the script."""
    module = load_module(script_path, name=pipeline_type)
    return getattr(module, method_name)


def get_method_params(method: Any) -> list[str]:
    """Return the parameters of a method."""
    signature = inspect.signature(method)
    return list(signature.parameters.keys())


def get_ingredients(
    method_params: list[str],
    reserved_params: list[str],
    passed_ingredients: dict[str, Any],
) -> dict[str, Any]:
    """Return ingredients for the given method params."""
    ingredients = {}
    for method_param in method_params:
        if method_param in reserved_params or method_param in ["kwargs", "_"]:
            continue
        if method_param not in passed_ingredients:
            raise ValueError(
                f"method param '{method_param}' doesn't exist in the ingredients"
            )
        ingredients[method_param] = passed_ingredients[method_param]

    return ingredients


class BasePipeline(ABC):
    """Base class for all pipelines."""

    recipe_name: str
    script_path: str
    method_name: str
    _method: Any
    _method_params: list[str]
    _passed_ingredients: dict[str, Any]
    ingredients: dict[str, Any]
    datasets: list[BaseDataset]

    @property
    @abstractmethod
    def pipeline_type(self) -> str:
        """Type of pipeline (ingest, query, cleanup)."""

    @property
    @abstractmethod
    def get_reserved_params(self) -> list[str]:
        """Get the list of reserved parameter names for this pipeline type."""

    def __init__(
        self,
        recipe_name: str,
        script_path: str,
        method_name: str,
        ingredients: dict[str, Any],
        datasets: list[BaseDataset],
    ):
        self.recipe_name = recipe_name
        self.script_path = script_path
        self.method_name = method_name
        self._passed_ingredients = ingredients
        self.datasets = datasets

        try:
            self._method = get_method(
                script_path=self.script_path,
                pipeline_type=self.pipeline_type,
                method_name=self.method_name,
            )
            self._method_params = get_method_params(method=self._method)
            self.ingredients = get_ingredients(
                method_params=self._method_params,
                reserved_params=self.get_reserved_params,
                passed_ingredients=self._passed_ingredients,
            )
        except BaseException:  # noqa: BLE001
            logging.critical(
                "Issue loading recipe %s on %s/%s with passed ingredients: %s",
                self.recipe_name,
                self.script_path,
                self.method_name,
                self._passed_ingredients,
                exc_info=True,
            )
            sys.exit(1)

    def get_method(self) -> Any:
        """Return the pipeline method."""
        return self._method

    def dataset_names(self) -> list[str]:
        """Return the names of the datasets."""
        return [d.name for d in self.datasets]

    def key(self) -> str:
        """Return the pipeline key."""
        key_parts = [
            self.pipeline_type,
            self.script_path,
            self.method_name,
        ]
        for name, value in self.ingredients.items():
            key_parts.append(f"{name}_{value}")
        return "_".join(key_parts)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BasePipeline):
            return self.key() == other.key()
        return False

    def __hash__(self) -> int:
        return hash(self.key())
