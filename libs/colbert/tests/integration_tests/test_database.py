import pytest
from cassandra.cluster import Session
from ragstack_colbert import CassandraDatabase, Chunk
from ragstack_tests_utils import TestData


@pytest.mark.parametrize("session", ["cassandra", "astra_db"], indirect=["session"])
def test_database_sync(session: Session):
    doc_id = "earth_doc_id"

    chunk_0 = Chunk(
        doc_id=doc_id,
        chunk_id=0,
        text=TestData.climate_change_text(),
        metadata={"name": "climate_change", "id": 23},
        embedding=TestData.climate_change_embedding(),
    )

    chunk_1 = Chunk(
        doc_id=doc_id,
        chunk_id=1,
        text=TestData.renewable_energy_text(),
        metadata={"name": "renewable_energy", "id": 42},
        embedding=TestData.renewable_energy_embedding(),
    )

    database = CassandraDatabase.from_session(
        keyspace="default_keyspace",
        table_name="test_database_sync",
        session=session,
    )

    results = database.add_chunks(chunks=[chunk_0, chunk_1])

    assert len(results) == 2  # noqa: PLR2004
    assert results[0] == (doc_id, 0)
    assert results[1] == (doc_id, 1)

    # TODO: verify other db methods.

    result = database.delete_chunks(doc_ids=[doc_id])
    assert result


@pytest.mark.parametrize("session", ["cassandra", "astra_db"], indirect=["session"])
async def test_database_async(session: Session):
    doc_id = "earth_doc_id"

    chunk_0 = Chunk(
        doc_id=doc_id,
        chunk_id=0,
        text=TestData.climate_change_text(),
        metadata={"name": "climate_change", "id": 23},
        embedding=TestData.climate_change_embedding(),
    )

    chunk_1 = Chunk(
        doc_id=doc_id,
        chunk_id=1,
        text=TestData.renewable_energy_text(),
        metadata={"name": "renewable_energy", "id": 42},
        embedding=TestData.renewable_energy_embedding(),
    )

    database = CassandraDatabase.from_session(
        keyspace="default_keyspace",
        table_name="test_database_async",
        session=session,
    )

    results = await database.aadd_chunks(chunks=[chunk_0, chunk_1])
    assert len(results) == 2  # noqa: PLR2004
    assert results[0] == (doc_id, 0)
    assert results[1] == (doc_id, 1)

    chunks = await database.search_relevant_chunks(vector=chunk_0.embedding[5], n=2)
    assert len(chunks) == 1
    assert chunks[0].doc_id == doc_id
    assert chunks[0].chunk_id == 0
    assert chunks[0].text is None
    assert chunks[0].metadata == {}
    assert chunks[0].embedding is None

    chunk = await database.get_chunk_embedding(doc_id=doc_id, chunk_id=1)
    assert chunk.doc_id == doc_id
    assert chunk.chunk_id == 1
    assert chunk.text is None
    assert chunk.metadata == {}
    assert chunk.embedding == chunk_1.embedding

    chunk = await database.get_chunk_data(doc_id=doc_id, chunk_id=0)
    assert chunk.doc_id == doc_id
    assert chunk.chunk_id == 0
    assert chunk.text == chunk_0.text
    # this is broken due to a cassio bug
    # which converts Number fields to strings
    # assert chunk.metadata == chunk_0.metadata
    assert chunk.embedding is None

    chunk = await database.get_chunk_data(
        doc_id=doc_id, chunk_id=0, include_embedding=True
    )
    assert chunk.doc_id == doc_id
    assert chunk.chunk_id == 0
    assert chunk.text == chunk_0.text
    # this is broken due to a cassio bug
    # which converts Number fields to strings
    # assert chunk.metadata == chunk_0.metadata
    assert chunk.embedding == chunk_0.embedding

    result = await database.adelete_chunks(doc_ids=[doc_id])
    assert result
