from typing import Any, Dict, Set
from ragstack_knowledge_store.graph_store import _serialize_metadata, _deserialize_metadata, _serialize_links, _deserialize_links
from ragstack_knowledge_store.links import Link

def test_metadata_serialization():
    def assert_roundtrip(metadata: Dict[str, Any]):
        serialized = _serialize_metadata(metadata)
        deserialized = _deserialize_metadata(serialized)
        assert metadata == deserialized

    assert_roundtrip({})
    assert_roundtrip({
        "a": "hello",
        "b": ["c", "d"],
        "c": []
    })

def test_links_serialization():
    def assert_roundtrip(links: Set[Link]):
        serialized = _serialize_links(links)
        deserialized = _deserialize_links(serialized)
        assert links == deserialized

    assert_roundtrip(set())
    assert_roundtrip({
        Link.incoming("a", "b"),
        Link.outgoing("a", "b"),
    })
    assert_roundtrip({
        Link.bidir("a", "b")
    })