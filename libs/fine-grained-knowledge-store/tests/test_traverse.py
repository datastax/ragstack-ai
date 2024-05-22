from precisely import assert_that, contains_exactly

from knowledge_graph.traverse import Node, Relation, atraverse, traverse

from .conftest import DataFixture


def test_traverse_empty(marie_curie: DataFixture) -> None:
    results = traverse(
        start=[],
        steps=1,
        edge_table=marie_curie.edge_table,
        session=marie_curie.session,
        keyspace=marie_curie.keyspace,
    )
    assert_that(results, contains_exactly())


def test_traverse_marie_curie(marie_curie: DataFixture) -> None:
    results = traverse(
        start=Node("Marie Curie", "Person"),
        steps=1,
        edge_table=marie_curie.edge_table,
        session=marie_curie.session,
        keyspace=marie_curie.keyspace,
    )
    expected = {
        Relation(Node("Marie Curie", "Person"), Node("Polish", "Nationality"), "HAS_NATIONALITY"),
        Relation(Node("Marie Curie", "Person"), Node("French", "Nationality"), "HAS_NATIONALITY"),
        Relation(
            Node("Marie Curie", "Person"), Node("Physicist", "Profession"), "HAS_PROFESSION"
        ),
        Relation(Node("Marie Curie", "Person"), Node("Chemist", "Profession"), "HAS_PROFESSION"),
        Relation(
            Node("Marie Curie", "Person"), Node("Professor", "Profession"), "HAS_PROFESSION"
        ),
        Relation(
            Node("Marie Curie", "Person"),
            Node("Radioactivity", "Scientific concept"),
            "RESEARCHED",
        ),
        Relation(Node("Marie Curie", "Person"), Node("Nobel Prize", "Award"), "WON"),
        Relation(Node("Marie Curie", "Person"), Node("Pierre Curie", "Person"), "MARRIED_TO"),
        Relation(
            Node("Marie Curie", "Person"),
            Node("University of Paris", "Organization"),
            "WORKED_AT",
        ),
    }
    assert_that(results, contains_exactly(*expected))

    results = traverse(
        start=Node("Marie Curie", "Person"),
        steps=2,
        edge_table=marie_curie.edge_table,
        session=marie_curie.session,
        keyspace=marie_curie.keyspace,
    )
    expected.add(Relation(Node("Pierre Curie", "Person"), Node("Nobel Prize", "Award"), "WON"))
    assert_that(results, contains_exactly(*expected))


async def test_atraverse_empty(marie_curie: DataFixture) -> None:
    results = await atraverse(
        start=[],
        steps=1,
        edge_table=marie_curie.edge_table,
        session=marie_curie.session,
        keyspace=marie_curie.keyspace,
    )
    assert_that(results, contains_exactly())


async def test_atraverse_marie_curie(marie_curie: DataFixture) -> None:
    results = await atraverse(
        start=Node("Marie Curie", "Person"),
        steps=1,
        edge_table=marie_curie.edge_table,
        session=marie_curie.session,
        keyspace=marie_curie.keyspace,
    )
    expected = {
        Relation(Node("Marie Curie", "Person"), Node("Polish", "Nationality"), "HAS_NATIONALITY"),
        Relation(Node("Marie Curie", "Person"), Node("French", "Nationality"), "HAS_NATIONALITY"),
        Relation(
            Node("Marie Curie", "Person"), Node("Physicist", "Profession"), "HAS_PROFESSION"
        ),
        Relation(Node("Marie Curie", "Person"), Node("Chemist", "Profession"), "HAS_PROFESSION"),
        Relation(
            Node("Marie Curie", "Person"), Node("Professor", "Profession"), "HAS_PROFESSION"
        ),
        Relation(
            Node("Marie Curie", "Person"),
            Node("Radioactivity", "Scientific concept"),
            "RESEARCHED",
        ),
        Relation(Node("Marie Curie", "Person"), Node("Nobel Prize", "Award"), "WON"),
        Relation(Node("Marie Curie", "Person"), Node("Pierre Curie", "Person"), "MARRIED_TO"),
        Relation(
            Node("Marie Curie", "Person"),
            Node("University of Paris", "Organization"),
            "WORKED_AT",
        ),
    }
    assert_that(results, contains_exactly(*expected))

    results = await atraverse(
        start=Node("Marie Curie", "Person"),
        steps=2,
        edge_table=marie_curie.edge_table,
        session=marie_curie.session,
        keyspace=marie_curie.keyspace,
    )
    expected.add(Relation(Node("Pierre Curie", "Person"), Node("Nobel Prize", "Award"), "WON"))
    assert_that(results, contains_exactly(*expected))


def test_traverse_marie_curie_filtered_edges(marie_curie: DataFixture) -> None:
    results = traverse(
        start=Node("Marie Curie", "Person"),
        steps=1,
        edge_filters=["edge_type = 'HAS_NATIONALITY'"],
        edge_table=marie_curie.edge_table,
        session=marie_curie.session,
        keyspace=marie_curie.keyspace,
    )
    expected = {
        Relation(Node("Marie Curie", "Person"), Node("Polish", "Nationality"), "HAS_NATIONALITY"),
        Relation(Node("Marie Curie", "Person"), Node("French", "Nationality"), "HAS_NATIONALITY"),
    }
    assert_that(results, contains_exactly(*expected))


async def test_atraverse_marie_curie_filtered_edges(marie_curie: DataFixture) -> None:
    results = await atraverse(
        start=Node("Marie Curie", "Person"),
        steps=1,
        edge_filters=["edge_type = 'HAS_NATIONALITY'"],
        edge_table=marie_curie.edge_table,
        session=marie_curie.session,
        keyspace=marie_curie.keyspace,
    )
    expected = {
        Relation(Node("Marie Curie", "Person"), Node("Polish", "Nationality"), "HAS_NATIONALITY"),
        Relation(Node("Marie Curie", "Person"), Node("French", "Nationality"), "HAS_NATIONALITY"),
    }
    assert_that(results, contains_exactly(*expected))
