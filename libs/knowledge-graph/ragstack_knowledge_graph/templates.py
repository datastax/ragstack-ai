from os import path
from typing import Callable, Union, cast

from langchain_core.prompts import PromptTemplate

TEMPLATE_PATH = path.join(path.dirname(__file__), "prompt_templates")


def load_template(
    filename: str, **kwargs: Union[str, Callable[[], str]]
) -> PromptTemplate:
    """Load a template from a file."""
    template = PromptTemplate.from_file(path.join(TEMPLATE_PATH, filename))
    if kwargs:
        template = cast(PromptTemplate, template.partial(**kwargs))
    return template
