from os import path
from typing import Callable, Union

from langchain_core.prompts import PromptTemplate

TEMPLATE_PATH = path.join(path.dirname(__file__), "prompt_templates")


def load_template(filename: str, **kwargs: Union[str, Callable[[], str]]) -> PromptTemplate:
    template = PromptTemplate.from_file(path.join(TEMPLATE_PATH, filename))
    if kwargs:
        template = template.partial(**kwargs)
    return template
