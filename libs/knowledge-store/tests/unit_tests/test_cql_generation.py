from ragstack_knowledge_store.graph_store import GraphStore, MetadataIndexingMode


class FakePreparedStatement:
    query_string: str

    def __init__(self, query: str) -> None:
        self.query_string = query


class FakeSession:
    def prepare(self, query: str) -> FakePreparedStatement:
        return FakePreparedStatement(query=query)


def _normalize_whitespace(s: str) -> str:
    return " ".join(s.split())


def test_cql_generation() -> None:
    gs = object.__new__(GraphStore)

    gs._keyspace = "test_keyspace"  # noqa: SLF001
    gs._node_table = "test_table"  # noqa: SLF001
    gs._session = FakeSession()  # noqa: SLF001
    gs._prepared_query_cache = {}  # noqa: SLF001

    query, values = gs._get_search_cql_and_params(limit=2, embedding=[0, 1])  # noqa: SLF001
    assert _normalize_whitespace(query.query_string) == _normalize_whitespace("""
        SELECT content_id, kind, text_content, links_blob, metadata_blob
        FROM test_keyspace.test_table
        ORDER BY text_embedding ANN OF ?
        LIMIT ?;
    """)
    assert values == ([0, 1], 2)

    query, values = gs._get_search_cql_and_params(  # noqa: SLF001
        limit=2, embedding=[0, 1], columns="content_id, link_to_tags"
    )
    assert _normalize_whitespace(query.query_string) == _normalize_whitespace("""
        SELECT content_id, link_to_tags
        FROM test_keyspace.test_table
        ORDER BY text_embedding ANN OF ?
        LIMIT ?;
    """)
    assert values == ([0, 1], 2)

    query, values = gs._get_search_cql_and_params(  # noqa: SLF001
        limit=2, embedding=[0, 1], columns="content_id, text_embedding, link_to_tags"
    )
    assert _normalize_whitespace(query.query_string) == _normalize_whitespace("""
        SELECT content_id, text_embedding, link_to_tags
        FROM test_keyspace.test_table
        ORDER BY text_embedding ANN OF ?
        LIMIT ?;
    """)
    assert values == ([0, 1], 2)

    query, values = gs._get_search_cql_and_params(  # noqa: SLF001
        columns="content_id AS target_content_id", link_from_tags=("link", "tag")
    )
    assert _normalize_whitespace(query.query_string) == _normalize_whitespace("""
        SELECT content_id AS target_content_id
        FROM test_keyspace.test_table
        WHERE link_from_tags CONTAINS (?, ?);
    """)
    assert values == ("link", "tag")

    columns = """
        content_id AS target_content_id,
        text_embedding AS target_text_embedding,
        link_to_tags AS target_link_to_tags
    """
    query, values = gs._get_search_cql_and_params(  # noqa: SLF001
        limit=2, embedding=[0, 1], columns=columns, link_from_tags=("link", "tag")
    )
    assert _normalize_whitespace(query.query_string) == _normalize_whitespace("""
        SELECT
            content_id AS target_content_id,
            text_embedding AS target_text_embedding,
            link_to_tags AS target_link_to_tags
        FROM test_keyspace.test_table
        WHERE link_from_tags CONTAINS (?, ?)
        ORDER BY text_embedding ANN OF ?
        LIMIT ?;
    """)
    assert values == ("link", "tag", [0, 1], 2)


def test_cql_generation_with_metadata() -> None:
    gs = object.__new__(GraphStore)

    gs._keyspace = "test_keyspace"  # noqa: SLF001
    gs._node_table = "test_table"  # noqa: SLF001
    gs._session = FakeSession()  # noqa: SLF001
    gs._prepared_query_cache = {}  # noqa: SLF001
    gs._metadata_indexing_policy = (MetadataIndexingMode.DEFAULT_TO_SEARCHABLE, set())  # noqa: SLF001

    query, values = gs._get_search_cql_and_params(  # noqa: SLF001
        limit=2, embedding=[0, 1], metadata={"one": True, "two": 2}
    )
    assert _normalize_whitespace(query.query_string) == _normalize_whitespace("""
        SELECT content_id, kind, text_content, links_blob, metadata_blob
        FROM test_keyspace.test_table
        WHERE metadata_s['one'] = ? AND metadata_s['two'] = ?
        ORDER BY text_embedding ANN OF ?
        LIMIT ?;
    """)
    assert values == ("true", "2.0", [0, 1], 2)

    query, values = gs._get_search_cql_and_params(  # noqa: SLF001
        limit=2,
        embedding=[0, 1],
        columns="content_id, link_to_tags",
        metadata={"three": "four"},
    )
    assert _normalize_whitespace(query.query_string) == _normalize_whitespace("""
        SELECT content_id, link_to_tags
        FROM test_keyspace.test_table
        WHERE metadata_s['three'] = ?
        ORDER BY text_embedding ANN OF ?
        LIMIT ?;
    """)
    assert values == ("four", [0, 1], 2)

    query, values = gs._get_search_cql_and_params(  # noqa: SLF001
        limit=2,
        embedding=[0, 1],
        columns="content_id, text_embedding, link_to_tags",
        metadata={"test": False},
    )
    assert _normalize_whitespace(query.query_string) == _normalize_whitespace("""
        SELECT content_id, text_embedding, link_to_tags
        FROM test_keyspace.test_table
        WHERE metadata_s['test'] = ?
        ORDER BY text_embedding ANN OF ?
        LIMIT ?;
    """)
    assert values == ("false", [0, 1], 2)

    query, values = gs._get_search_cql_and_params(  # noqa: SLF001
        columns="content_id AS target_content_id",
        link_from_tags=("link", "tag"),
        metadata={"one": True, "two": 2},
    )
    assert _normalize_whitespace(query.query_string) == _normalize_whitespace("""
        SELECT content_id AS target_content_id
        FROM test_keyspace.test_table
        WHERE link_from_tags CONTAINS (?, ?) AND metadata_s['one'] = ?
            AND metadata_s['two'] = ?;
    """)
    assert values == ("link", "tag", "true", "2.0")

    columns = """
        content_id AS target_content_id,
        text_embedding AS target_text_embedding,
        link_to_tags AS target_link_to_tags
    """
    query, values = gs._get_search_cql_and_params(  # noqa: SLF001
        limit=2,
        embedding=[0, 1],
        columns=columns,
        link_from_tags=("link", "tag"),
        metadata={"five": "3.0"},
    )
    assert _normalize_whitespace(query.query_string) == _normalize_whitespace("""
        SELECT
            content_id AS target_content_id,
            text_embedding AS target_text_embedding,
            link_to_tags AS target_link_to_tags
        FROM test_keyspace.test_table
        WHERE link_from_tags CONTAINS (?, ?) AND metadata_s['five'] = ?
        ORDER BY text_embedding ANN OF ?
        LIMIT ?;
    """)
    assert values == ("link", "tag", "3.0", [0, 1], 2)
