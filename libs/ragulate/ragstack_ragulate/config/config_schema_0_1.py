from typing import Any, Dict

from ragstack_ragulate.datasets import BaseDataset, find_dataset, get_dataset

from .base_config_schema import BaseConfigSchema
from .objects import Config, Recipe, Step
from .utils import dict_to_string


class ConfigSchema_0_1(BaseConfigSchema):
    def version(self):
        return 0.1

    def schema(self) -> Dict[str, Any]:
        step_list = {
            "type": "list",
            "schema": {
                "type": "dict",
                "schema": {
                    "name": {"type": "string"},
                    "script": {"type": "string"},
                    "method": {"type": "string"},
                },
            },
        }

        steps = {
            "type": "dict",
            "schema": {
                "ingest": step_list,
                "query": step_list,
                "cleanup": step_list,
            },
        }

        ingredient_list = {
            "type": "list",
            "schema": {
                "type": "dict",
                "allow_unknown": True,
                "minlength": 1,
                "maxlength": 1,
            },
        }

        recipe_list = {
            "type": "list",
            "schema": {
                "type": "dict",
                "schema": {
                    "name": {"type": "string"},
                    "ingest": {"type": "string"},
                    "query": {"type": "string"},
                    "cleanup": {"type": "string"},
                    "ingredients": ingredient_list,
                },
            },
        }

        datasets_kinds = ["llama"]

        dataset_list = {
            "type": "list",
            "oneof": [
                {"schema": {"type": "string"}},
                {
                    "schema": {
                        "type": "dict",
                        "schema": {
                            "name": {"type": "string"},
                            "kind": {"type": "string", "allowed": datasets_kinds},
                        },
                    }
                },
            ],
        }

        llm_vendors = ["openai", "open_ai", "huggingface", "hugging_face"]

        llm_list = {
            "type": "list",
            "oneof": [
                {"schema": {"type": "string"}},
                {
                    "schema": {
                        "type": "dict",
                        "schema": {
                            "name": {"type": "string"},
                            "vendor": {
                                "type": "string",
                                "required": True,
                                "allowed": llm_vendors,
                            },
                            "model": {"type": "string", "required": True},
                            "default": {"type": "boolean"},
                        },
                    }
                },
            ],
        }

        metric_options = {
            "type": "dict",
            "schema": {
                "enabled": {"type": "boolean"},
                "eval_llm": {"type": "string"},
            },
        }

        supported_metrics = [
            "answer_correctness",
            "answer_relevance",
            "context_relevance",
            "groundedness",
        ]

        metrics = {
            "oneof": [
                {"type": "list", "allowed": supported_metrics},
                {
                    "type": "dict",
                    "schema": {metric: metric_options for metric in supported_metrics},
                },
            ]
        }

        schema = {
            "version": {"type": "float", "allowed": [0.1]},
            "steps": steps,
            "recipes": recipe_list,
            "datasets": dataset_list,
            "eval_llms": llm_list,
            "metrics": metrics,
        }

        return schema

    def parse_document(self, document: Dict[str, Any]) -> Config:
        ingest_steps: Dict[str, Step] = {}
        query_steps: Dict[str, Step] = {}
        cleanup_steps: Dict[str, Step] = {}

        step_map = {
            "ingest": ingest_steps,
            "query": query_steps,
            "cleanup": cleanup_steps,
        }

        doc_steps = document.get("steps", {})
        for step_kind, steps in step_map.items():
            for doc_step in doc_steps.get(step_kind, {}):
                doc_name = doc_step.get("name", None)
                doc_script = doc_step.get("script", None)
                doc_method = doc_step.get("method", None)
                if doc_name in steps:
                    raise ValueError(
                        f"{step_kind} step names must be unique. Found {doc_name} more than once."
                    )
                steps[doc_name] = Step(
                    name=doc_name, script=doc_script, method=doc_method
                )

        recipes: Dict[str, Recipe] = {}

        doc_recipes = document.get("recipes", {})
        for doc_recipe in doc_recipes:
            doc_ingredients = doc_recipe.get("ingredients", {})
            ingredients: Dict[str, Any] = {}

            for doc_ingredient in doc_ingredients:
                for key, value in doc_ingredient.items():
                    if key in ingredients:
                        raise ValueError(
                            f"ingredient {key} appears in recipe more than once."
                        )
                    ingredients[key] = value

            doc_name = doc_recipe.get("name", None)

            if doc_name is None:
                if len(doc_ingredients) == 0:
                    raise ValueError(
                        f"recipe must either have a `name` defined or contain at least one ingredient."
                    )
                else:
                    recipe_name = dict_to_string(ingredients)
            else:
                recipe_name = doc_name

            recipe_steps: Dict[str, Step] = {}

            for step_kind in step_map:
                doc_recipe_step = doc_recipe.get(step_kind, None)
                step = step_map[step_kind].get(doc_recipe_step, None)
                if doc_recipe_step is not None and step is None:
                    raise ValueError(
                        f"{step_kind} step {doc_recipe_step} for recipe {recipe_name} is not defined in the `steps` section"
                    )
                else:
                    recipe_steps[step_kind] = step

            if "query" not in recipe_steps:
                raise ValueError(f"query step is missing for recipe {recipe_name}")

            if recipe_name in recipes:
                raise ValueError(
                    f"recipe names must be unique. Found {recipe_name} more than once."
                )

            recipes[recipe_name] = Recipe(
                name=recipe_name,
                ingest=recipe_steps.get("ingest"),
                query=recipe_steps.get("query"),
                cleanup=recipe_steps.get("cleanup"),
                ingredients=ingredients,
            )

            datasets: Dict[str, BaseDataset] = {}

            for doc_dataset in document.get("datasets", []):
                if isinstance(doc_dataset, str):
                    datasets[doc_dataset] = find_dataset(name=doc_dataset)
                else:
                    doc_dataset_name = doc_dataset.get("name", None)
                    doc_dataset_kind = doc_dataset.get("kind", None)
                    if doc_dataset_name is None or doc_dataset_kind is None:
                        raise ValueError(
                            "datasets must be specified with `name` and `kind`"
                        )
                    datasets[doc_dataset_name] = get_dataset(
                        name=doc_dataset_name, kind=doc_dataset_kind
                    )

        return Config(recipes=recipes, datasets=datasets)
