import pytest
from langchain_core.documents import Document

from ragstack_knowledge_store.base import _documents_to_nodes, _texts_to_nodes, TextNode
from .conftest import DataFixture


def test_write_retrieve_href_url_pair(fresh_fixture: DataFixture):
    a = Document(
        page_content="A",
        metadata={
            "content_id": "a",
            "urls": ["http://a"],
        },
    )
    b = Document(
        page_content="B",
        metadata={
            "content_id": "b",
            "hrefs": ["http://a"],
            "urls": ["http://b"],
        },
    )
    c = Document(
        page_content="C",
        metadata={
            "content_id": "c",
            "hrefs": ["http://a"],
        },
    )
    d = Document(
        page_content="D",
        metadata={"content_id": "d", "hrefs": ["http://a", "http://b"]},
    )

    store = fresh_fixture.store([a, b, c, d])

    assert list(store._linked_ids("a")) == []
    assert list(store._linked_ids("b")) == ["a"]
    assert list(store._linked_ids("c")) == ["a"]
    assert sorted(store._linked_ids("d")) == ["a", "b"]


def test_write_retrieve_keywords(fresh_fixture: DataFixture):
    _texts_to_nodes(["a", "b"], {"a": "b"}, None)
    greetings = Document(
        page_content="Typical Greetings",
        metadata={
            "content_id": "greetings",
        },
    )
    doc1 = Document(
        page_content="Hello World",
        metadata={
            "content_id": "doc1",
            "parent_content_id": "greetings",
            "keywords": {"greeting", "world"},
        },
    )
    doc2 = Document(
        page_content="Hello Earth",
        metadata={
            "content_id": "doc2",
            "parent_content_id": "greetings",
            "keywords": {"greeting", "earth"},
        },
    )

    store = fresh_fixture.store([greetings, doc1, doc2])

    # Doc2 is more similar, but World and Earth are similar enough that doc1 also shows up.
    results = store.similarity_search("Earth", k=2)
    assert list(map(lambda d: d.page_content, results)) == [
        doc2.page_content,
        doc1.page_content,
    ]

    results = store.similarity_search("Earth", k=1)
    assert list(map(lambda d: d.page_content, results)) == [doc2.page_content]

    results = store.traversing_retrieve("Earth", k=2, depth=0)
    assert set(map(lambda d: d.page_content, results)) == {
        doc2.page_content,
        doc1.page_content,
    }

    results = store.traversing_retrieve("Earth", k=2, depth=1)
    assert set(map(lambda d: d.page_content, results)) == {
        doc2.page_content,
        doc1.page_content,
        greetings.page_content,
    }

    # K=1 only pulls in doc2 (Hello Earth)
    results = store.traversing_retrieve("Earth", k=1, depth=0)
    assert set(map(lambda d: d.page_content, results)) == {doc2.page_content}

    # K=1 only pulls in doc2 (Hello Earth). Depth=1 traverses to parent and via keyword edge.
    results = store.traversing_retrieve("Earth", k=1, depth=1)
    assert set(map(lambda d: d.page_content, results)) == {
        doc2.page_content,
        doc1.page_content,
        greetings.page_content,
    }


def test_texts_to_nodes():
    assert list(_texts_to_nodes(["a", "b"], [{"a": "b"}, {"c": "d"}], ["a", "b"])) == [
        TextNode(id="a", metadata={"a": "b"}, text="a"),
        TextNode(id="b", metadata={"c": "d"}, text="b"),
    ]
    assert list(_texts_to_nodes(["a", "b"], None, ["a", "b"])) == [
        TextNode(id="a", metadata={}, text="a"),
        TextNode(id="b", metadata={}, text="b"),
    ]
    assert list(_texts_to_nodes(["a", "b"], [{"a": "b"}, {"c": "d"}], None)) == [
        TextNode(metadata={"a": "b"}, text="a"),
        TextNode(metadata={"c": "d"}, text="b"),
    ]
    with pytest.raises(ValueError):
        list(_texts_to_nodes(["a", "b"], None, ["a"]))
    with pytest.raises(ValueError):
        list(_texts_to_nodes(["a", "b"], [{"a": "b"}], None))
    with pytest.raises(ValueError):
        list(_texts_to_nodes(["a"], [{"a": "b"}, {"c": "d"}], None))
    with pytest.raises(ValueError):
        list(_texts_to_nodes(["a"], None, ["a", "b"]))


def test_documents_to_nodes():
    documents = [
        Document(page_content="a", metadata={"a": "b"}),
        Document(page_content="b", metadata={"c": "d"}),
    ]
    assert list(_documents_to_nodes(documents, ["a", "b"])) == [
        TextNode(id="a", metadata={"a": "b"}, text="a"),
        TextNode(id="b", metadata={"c": "d"}, text="b"),
    ]
    assert list(_documents_to_nodes(documents, None)) == [
        TextNode(metadata={"a": "b"}, text="a"),
        TextNode(metadata={"c": "d"}, text="b"),
    ]
    with pytest.raises(ValueError):
        list(_documents_to_nodes(documents, ["a"]))
    with pytest.raises(ValueError):
        list(_documents_to_nodes(documents[1:], ["a", "b"]))
