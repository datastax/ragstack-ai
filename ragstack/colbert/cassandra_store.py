import logging
from typing import List

from cassandra.cluster import Session
from cassandra.concurrent import execute_concurrent_with_args

from .token_embedding import EmbeddedChunk
from .vector_store import ColbertVectorStore


class CassandraColbertVectorStore(ColbertVectorStore):
    session: Session
    keyspace: str
    table_name: str

    def __init__(self, session: Session, keyspace: str, table_name: str):
        self.keyspace = keyspace
        self.session = session
        self.table_name = table_name
        self.full_table_name = f"{keyspace}.{table_name}"
        self.__create_tables()

        # prepare statements
        self.insert_chunk_stmt = self.session.prepare(
            f"""
        INSERT INTO {self.full_table_name} (doc_id, chunk_id, embedding_id, body)
        VALUES (?, ?, -1, ?)
        """
        )

        self.insert_colbert_stmt = self.session.prepare(
            f"""
        INSERT INTO {self.full_table_name} (doc_id, chunk_id, embedding_id, bert_embedding)
        VALUES (?, ?, ?, ?)
        """
        )

        self.query_colbert_ann_stmt = self.session.prepare(
            f"""
        SELECT doc_id, chunk_id
        FROM {self.full_table_name}
        ORDER BY bert_embedding ANN OF ?
        LIMIT ?
        """
        )

        self.query_colbert_parts_stmt = self.session.prepare(
            f"""
        SELECT doc_id, chunk_id, bert_embedding
        FROM {self.full_table_name}
        WHERE doc_id = ? AND chunk_id = ? AND embedding_id != -1
        """
        )

        self.query_chunk_stmt = self.session.prepare(
            f"""
        SELECT body
        FROM {self.full_table_name}
        WHERE doc_id = ? AND chunk_id = ? AND embedding_id = -1
        """
        )

        self.delete_part_by_id_stmt = self.session.prepare(
            f"""
            DELETE FROM {self.full_table_name} WHERE doc_id = ?
        """
        )

    def __create_tables(self) -> None:
        self.session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.full_table_name} (
                doc_id text,
                chunk_id int,
                embedding_id int,
                body text,
                bert_embedding vector<float, 128>,
                PRIMARY KEY (doc_id, chunk_id, embedding_id)
            ) WITH COMMENT = 'Colbert embeddings embedding_id=-1 contains the body chunk';
        """
        )
        logging.info(f"Created table {self.full_table_name}")

        self.session.execute(
            f"""
            CREATE CUSTOM INDEX IF NOT EXISTS colbert_ann_{self.table_name} ON {self.full_table_name}(bert_embedding) USING 'StorageAttachedIndex'
  WITH OPTIONS = {{'similarity_function': 'DOT_PRODUCT' }};
        """
        )
        logging.info(f"Created index on table {self.full_table_name}")

    def insert_colbert_embedding_docs(
        self, chunks: List[EmbeddedChunk], delete_existing: bool = False
    ) -> None:
        if delete_existing:
            doc_ids = [c.doc_id() for c in chunks]
            self.delete_documents(list(set(doc_ids)))

        p_parameters = [(c.doc_id(), c.chunk_id(), c.text()) for c in chunks]
        execute_concurrent_with_args(self.session, self.insert_chunk_stmt, p_parameters)

        for chunk in chunks:
            doc_id = chunk.doc_id()
            chunk_id = chunk.chunk_id()
            parameters = [
                (doc_id, chunk_id, index, vector)
                for index, vector in enumerate(chunk.vectors().tolist())
            ]
            execute_concurrent_with_args(
                self.session, self.insert_colbert_stmt, parameters
            )

    def put_chunks(
        self, chunks: List[EmbeddedChunk], delete_existing: bool = False
    ) -> None:
        return self.insert_colbert_embedding_docs(chunks, delete_existing)

    def delete_documents(self, doc_ids: List[str]) -> None:
        execute_concurrent_with_args(
            self.session, self.delete_part_by_id_stmt, [(t,) for t in doc_ids]
        )

    def close(self) -> None:
        pass
