# ruff: noqa: SLF001
"""
Stringification of everything in the simple metadata handling
"""

from ragstack_knowledge_store.graph_store import GraphStore


class TestMetadataStringCoercion:
    def test_metadata_string_coercion(self) -> None:
        md_dict = {
            "integer": 1,
            "float": 2.0,
            "boolean": True,
            "null": None,
            "string": "letter E",
            "something": RuntimeError("You cannot do this!"),
        }

        stringified = {k: GraphStore._coerce_string(v) for k, v in md_dict.items()}

        expected = {
            "integer": "1.0",
            "float": "2.0",
            "boolean": "true",
            "null": "null",
            "string": "letter E",
            "something": str(RuntimeError("You cannot do this!")),
        }

        assert stringified == expected
