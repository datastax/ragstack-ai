import pytest
from cassandra.cluster import Session
from ragstack_colbert import CassandraDatabase, Chunk
from ragstack_tests_utils import TestData


@pytest.mark.parametrize("session", ["cassandra", "astra_db"], indirect=["session"])
def test_database_sync(session: Session) -> None:
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

    assert results == [(doc_id, 0), (doc_id, 1)]

    # TODO: verify other db methods.

    result = database.delete_chunks(doc_ids=[doc_id])
    assert result


@pytest.mark.parametrize("session", ["cassandra", "astra_db"], indirect=["session"])
async def test_database_async(session: Session) -> None:
    doc_id = "earth_doc_id"

    climate_change_embedding = TestData.climate_change_embedding()

    chunk_0 = Chunk(
        doc_id=doc_id,
        chunk_id=0,
        text=TestData.climate_change_text(),
        metadata={"name": "climate_change", "id": 23},
        embedding=climate_change_embedding,
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
    assert results == [(doc_id, 0), (doc_id, 1)]

    chunks = await database.search_relevant_chunks(
        vector=climate_change_embedding[5], n=2
    )
    assert chunks == [
        Chunk(
            doc_id=doc_id,
            chunk_id=0,
            embedding=None,
        )
    ]

    chunk = await database.get_chunk_embedding(doc_id=doc_id, chunk_id=1)
    assert chunk == Chunk(
        doc_id=doc_id,
        chunk_id=1,
        embedding=chunk_1.embedding,
    )

    chunk = await database.get_chunk_data(doc_id=doc_id, chunk_id=0)

    assert chunk == Chunk(
        doc_id=doc_id,
        chunk_id=0,
        text=chunk_0.text,
        # this is broken due to a cassio bug
        # which converts Number fields to strings
        # metadata=chunk_0.metadata,
        embedding=None,
    )

    chunk = await database.get_chunk_data(
        doc_id=doc_id, chunk_id=0, include_embedding=True
    )

    assert chunk == Chunk(
        doc_id=doc_id,
        chunk_id=0,
        text=chunk_0.text,
        # this is broken due to a cassio bug
        # which converts Number fields to strings
        # metadata=chunk_0.metadata,
        embedding=chunk_0.embedding,
    )

    result = await database.adelete_chunks(doc_ids=[doc_id])
    assert result
