from langchain_community.graph_vectorstores.extractors import (
    GLiNERLinkExtractor,
    HtmlLinkExtractor,
    KeybertLinkExtractor,
)
from langchain_core.documents import Document
from langchain_core.graph_vectorstores.links import Link, get_links
from ragstack_langchain.graph_store.extractors import LinkExtractorTransformer

from . import (
    test_gliner_link_extractor,
    test_keybert_link_extractor,
)

PAGE_1 = """
<html>
<body>
Hello.
<a href="relative">Relative</a>
<a href="/relative-base">Relative base.</a>
<a href="http://cnn.com">Aboslute</a>
<a href="//same.foo">Test</a>
</body>
</html>
"""

PAGE_2 = """
<html>
<body>
Hello.
<a href="/bar/#fragment">Relative</a>
</html>
"""


def test_html_extractor():
    transformer = LinkExtractorTransformer(
        [
            HtmlLinkExtractor().as_document_extractor(),
        ]
    )
    doc1 = Document(
        page_content=PAGE_1,
        metadata={
            "source": "https://foo.com/bar/",
        },
    )
    doc2 = Document(
        page_content=PAGE_2,
        metadata={
            "source": "https://foo.com/baz/",
        },
    )
    results = transformer.transform_documents([doc1, doc2])
    assert results[0] == doc1
    assert results[1] == doc2

    assert set(get_links(doc1)) == {
        Link.incoming(kind="hyperlink", tag="https://foo.com/bar/"),
        Link.outgoing(kind="hyperlink", tag="https://foo.com/bar/relative"),
        Link.outgoing(kind="hyperlink", tag="https://foo.com/relative-base"),
        Link.outgoing(kind="hyperlink", tag="http://cnn.com"),
        Link.outgoing(kind="hyperlink", tag="https://same.foo"),
    }

    assert set(get_links(doc2)) == {
        Link.incoming(kind="hyperlink", tag="https://foo.com/baz/"),
        Link.outgoing(kind="hyperlink", tag="https://foo.com/bar/"),
    }


def test_multiple_extractors():
    transformer = LinkExtractorTransformer(
        [
            KeybertLinkExtractor(),
            GLiNERLinkExtractor(
                labels=["Person", "Award", "Date", "Competitions", "Teams"]
            ),
        ]
    )

    doc1 = Document(page_content=test_keybert_link_extractor.PAGE_1)
    doc2 = Document(page_content=test_gliner_link_extractor.PAGE_1)

    results = transformer.transform_documents([doc1, doc2])
    assert results[0] == doc1
    assert results[1] == doc2

    assert set(get_links(doc1)) == {
        Link(kind="kw", direction="bidir", tag="labeled"),
        Link(kind="kw", direction="bidir", tag="learning"),
        Link(kind="kw", direction="bidir", tag="training"),
        Link(kind="kw", direction="bidir", tag="supervised"),
        Link(kind="kw", direction="bidir", tag="labels"),
    }

    assert set(get_links(doc2)) == {
        Link(kind="kw", direction="bidir", tag="cristiano"),
        Link(kind="kw", direction="bidir", tag="goalscorer"),
        Link(kind="kw", direction="bidir", tag="footballer"),
        Link(kind="entity:Teams", direction="bidir", tag="Portugal national team"),
        Link(kind="entity:Date", direction="bidir", tag="5 February 1985"),
        Link(
            kind="entity:Person",
            direction="bidir",
            tag="Cristiano Ronaldo dos Santos Aveiro",
        ),
        Link(kind="kw", direction="bidir", tag="santos"),
        Link(kind="kw", direction="bidir", tag="ronaldo"),
        Link(kind="entity:Award", direction="bidir", tag="European Golden Shoes"),
        Link(kind="entity:Award", direction="bidir", tag="Ballon d'Or"),
        Link(
            kind="entity:Competitions",
            direction="bidir",
            tag="UEFA European Championship",
        ),
        Link(
            kind="entity:Competitions", direction="bidir", tag="European\nChampionship"
        ),
        Link(
            kind="entity:Competitions", direction="bidir", tag="UEFA Champions Leagues"
        ),
        Link(kind="entity:Competitions", direction="bidir", tag="UEFA\nNations League"),
        Link(
            kind="entity:Award",
            direction="bidir",
            tag="UEFA Men's\nPlayer of the Year Awards",
        ),
    }
