from enum import Enum
from typing import Optional, Set

from langchain_core.pydantic_v1 import BaseModel, Field


class Kind(str, Enum):
    document = "document"
    """A root document (PDF, HTML, etc.).

    Document content should have `content_id == document_id` and `parent_id = None`.
    """

    section = "section"
    """A section or sub-section (depending on parent_id) of a document.

    Top-level sections should have `parent_id == document_id`, while
    sub-sections should use the enclosing section as the `parent_id`.
    """

    passage = "passage"
    """A text-passage within a document."""

    image = "image"
    """An image within a document."""

    table = "table"
    """A table within a document."""


class Content(BaseModel):
    source_id: str
    content_id: str
    parent_id: Optional[str] = None
    kind: Kind
    keywords: Set[str] = Field(default_factory=set)
    urls: Set[str] = Field(default_factory=set)
    links: Set[str] = Field(default_factory=set)
    text_content: Optional[str] = None
