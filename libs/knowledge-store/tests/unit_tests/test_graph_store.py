from __future__ import annotations

from typing import Any

from ragstack_knowledge_store.graph_store import (
    _deserialize_links,
    _deserialize_metadata,
    _serialize_links,
    _serialize_metadata,
)
from ragstack_knowledge_store.links import Link


def test_metadata_serialization() -> None:
    def assert_roundtrip(metadata: dict[str, Any]) -> None:
        serialized = _serialize_metadata(metadata)
        deserialized = _deserialize_metadata(serialized)
        assert metadata == deserialized

    assert_roundtrip({})
    assert_roundtrip({"a": "hello", "b": ["c", "d"], "c": []})


def test_links_serialization() -> None:
    def assert_roundtrip(links: set[Link]) -> None:
        serialized = _serialize_links(links)
        deserialized = _deserialize_links(serialized)
        assert links == deserialized

    assert_roundtrip(set())
    assert_roundtrip(
        {
            Link.incoming("a", "b"),
            Link.outgoing("a", "b"),
        }
    )
    assert_roundtrip({Link.bidir("a", "b")})
