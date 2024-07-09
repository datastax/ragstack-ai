import pytest
from langchain_core.documents import Document
from ragstack_langchain.graph_store.base import (
    Node,
    _documents_to_nodes,
    _texts_to_nodes,
)
from ragstack_langchain.graph_store.links import Link


def test_texts_to_nodes() -> None:
    assert list(_texts_to_nodes(["a", "b"], [{"a": "b"}, {"c": "d"}], ["a", "b"])) == [
        Node(id="a", metadata={"a": "b"}, text="a"),
        Node(id="b", metadata={"c": "d"}, text="b"),
    ]
    assert list(_texts_to_nodes(["a", "b"], None, ["a", "b"])) == [
        Node(id="a", metadata={}, text="a"),
        Node(id="b", metadata={}, text="b"),
    ]
    assert list(_texts_to_nodes(["a", "b"], [{"a": "b"}, {"c": "d"}], None)) == [
        Node(metadata={"a": "b"}, text="a"),
        Node(metadata={"c": "d"}, text="b"),
    ]
    assert list(
        _texts_to_nodes(
            ["a"],
            [{"links": {Link.incoming(kind="hyperlink", tag="http://b")}}],
            None,
        )
    ) == [Node(links={Link.incoming(kind="hyperlink", tag="http://b")}, text="a")]
    with pytest.raises(ValueError, match="texts iterable longer than ids"):
        list(_texts_to_nodes(["a", "b"], None, ["a"]))
    with pytest.raises(ValueError, match="texts iterable longer than metadatas"):
        list(_texts_to_nodes(["a", "b"], [{"a": "b"}], None))
    with pytest.raises(ValueError, match="metadatas iterable longer than texts"):
        list(_texts_to_nodes(["a"], [{"a": "b"}, {"c": "d"}], None))
    with pytest.raises(ValueError, match="ids iterable longer than texts"):
        list(_texts_to_nodes(["a"], None, ["a", "b"]))


def test_documents_to_nodes() -> None:
    documents = [
        Document(
            page_content="a",
            metadata={"links": {Link.incoming(kind="hyperlink", tag="http://b")}},
        ),
        Document(page_content="b", metadata={"c": "d"}),
    ]
    assert list(_documents_to_nodes(documents, ["a", "b"])) == [
        Node(
            id="a",
            metadata={},
            links={Link.incoming(kind="hyperlink", tag="http://b")},
            text="a",
        ),
        Node(id="b", metadata={"c": "d"}, text="b"),
    ]
    assert list(_documents_to_nodes(documents, None)) == [
        Node(links={Link.incoming(kind="hyperlink", tag="http://b")}, text="a"),
        Node(metadata={"c": "d"}, text="b"),
    ]
    with pytest.raises(ValueError, match="documents iterable longer than ids"):
        list(_documents_to_nodes(documents, ["a"]))
    with pytest.raises(ValueError, match="ids iterable longer than documents"):
        list(_documents_to_nodes(documents[1:], ["a", "b"]))
