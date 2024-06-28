import importlib.util
import inspect
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ragstack_ragulate.datasets import BaseDataset

from ..logging_config import logger


# Function to dynamically load a module
def load_module(file_path, name):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_method(script_path: str, pipeline_type: str, method_name: str):
    module = load_module(script_path, name=pipeline_type)
    return getattr(module, method_name)


def get_method_params(method: Any) -> List[str]:
    signature = inspect.signature(method)
    return signature.parameters.keys()


def get_ingredients(
    method_params: List[str],
    reserved_params: List[str],
    passed_ingredients: Dict[str, Any],
) -> Dict[str, Any]:
    ingredients = {}
    for method_param in method_params:
        if method_param in reserved_params or method_param in ["kwargs"]:
            continue
        if method_param not in passed_ingredients:
            raise ValueError(
                f"method param '{method_param}' doesn't exist in the ingredients"
            )
        ingredients[method_param] = passed_ingredients[method_param]

    return ingredients


class BasePipeline(ABC):
    recipe_name: str
    script_path: str
    method_name: str
    _method: Any
    _method_params: List[str]
    _passed_ingredients: Dict[str, Any]
    ingredients: Dict[str, Any]
    datasets: List[BaseDataset]

    @property
    @abstractmethod
    def PIPELINE_TYPE(self):
        """type of pipeline (ingest, query, cleanup)"""
        pass

    @property
    @abstractmethod
    def get_reserved_params(self) -> List[str]:
        """get the list of reserved parameter names for this pipeline type"""

    def __init__(
        self,
        recipe_name: str,
        script_path: str,
        method_name: str,
        ingredients: Dict[str, Any],
        datasets: List[BaseDataset],
        **kwargs,
    ):
        self.recipe_name = recipe_name
        self.script_path = script_path
        self.method_name = method_name
        self._passed_ingredients = ingredients
        self.datasets = datasets

        try:
            self._method = get_method(
                script_path=self.script_path,
                pipeline_type=self.PIPELINE_TYPE,
                method_name=self.method_name,
            )
            self._method_params = get_method_params(method=self._method)
            self.ingredients = get_ingredients(
                method_params=self._method_params,
                reserved_params=self.get_reserved_params,
                passed_ingredients=self._passed_ingredients,
            )
        except BaseException as e:
            logger.fatal(
                f"Issue loading recipe {self.recipe_name} on {self.script_path}/{self.method_name} with passed ingredients: {self._passed_ingredients}: {e}"
            )
            traceback.print_exc()
            exit(1)

    def get_method(self):
        return self._method

    def dataset_names(self) -> List[str]:
        return [d.name for d in self.datasets]

    def _key(self) -> str:
        key_parts = [
            self.PIPELINE_TYPE,
            self.script_path,
            self.method_name,
        ]
        for name, value in self.ingredients.items():
            key_parts.append(f"{name}_{value}")
        return "_".join(key_parts)

    def __eq__(self, other):
        if isinstance(other, BasePipeline):
            return self._key() == other._key()
        return False

    def __hash__(self):
        return hash(self._key())
