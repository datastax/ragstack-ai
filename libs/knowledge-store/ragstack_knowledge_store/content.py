from enum import Enum


class Kind(str, Enum):
    """The kind of content in a document."""

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
