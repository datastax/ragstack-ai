from typing import List
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra import InvalidRequest
from cassandra.concurrent import execute_concurrent_with_args
import logging

from .token_embedding import PassageEmbeddings


class AstraDB:
    def __init__(
        self,
        secure_connect_bundle: str = None,
        astra_token: str = None,
        keyspace: str = "colbert128",
        timeout: int = 60,
        **kwargs,
    ):

        if secure_connect_bundle is None or astra_token is None:
            self.cluster = Cluster(**kwargs)
        else:
            self.cluster = Cluster(
                cloud={"secure_connect_bundle": secure_connect_bundle},
                auth_provider=PlainTextAuthProvider("token", astra_token),
            )
        self.keyspace = keyspace
        self.session = self.cluster.connect()
        self.session.default_timeout = timeout

        logging.info(f"set up keyspace {keyspace}, tables and indexes...")

        if keyspace not in self.cluster.metadata.keyspaces.keys():
            # On Astra, the keyspace has to be created manually from the UI
            # make sure the role with the permission to create keyspace on Astra
            create_keyspace_query = f"""
                CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
                WITH replication = {{
                'class': 'SimpleStrategy',
                'replication_factor': '1'
                }};
                """
            self.session.execute(create_keyspace_query)
            logging.info(f"keyspace '{keyspace}' created")
        self.create_tables()

        # prepare statements

        insert_chunk_cql = f"""
        INSERT INTO {keyspace}.chunks (title, part, body)
        VALUES (?, ?, ?)
        """
        self.insert_chunk_stmt = self.session.prepare(insert_chunk_cql)

        insert_colbert_cql = f"""
        INSERT INTO {keyspace}.colbert_embeddings (title, part, embedding_id, bert_embedding)
        VALUES (?, ?, ?, ?)
        """
        self.insert_colbert_stmt = self.session.prepare(insert_colbert_cql)

        query_colbert_ann_cql = f"""
        SELECT title, part
        FROM {keyspace}.colbert_embeddings
        ORDER BY bert_embedding ANN OF ?
        LIMIT ?
        """
        self.query_colbert_ann_stmt = self.session.prepare(query_colbert_ann_cql)

        query_colbert_parts_cql = f"""
        SELECT title, part, bert_embedding
        FROM {keyspace}.colbert_embeddings
        WHERE title = ? AND part = ?
        """
        self.query_colbert_parts_stmt = self.session.prepare(query_colbert_parts_cql)

        query_part_by_pk = f"""
        SELECT body
        FROM {keyspace}.chunks
        WHERE title = ? AND part = ?
        """
        self.query_part_by_pk_stmt = self.session.prepare(query_part_by_pk)

        logging.info("statements are prepared")

    def create_tables(self):
        self.session.execute(
            f"""
            use {self.keyspace};
        """
        )
        logging.info(f"Using keyspace {self.keyspace}")

        self.session.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks(
                title text,
                part int,
                body text,
                PRIMARY KEY (title, part)
            );
        """
        )
        logging.info("Created chunks table")

        self.session.execute(
            """
            CREATE TABLE IF NOT EXISTS colbert_embeddings (
                title text,
                part int,
                embedding_id int,
                bert_embedding vector<float, 128>,
                PRIMARY KEY (title, part, embedding_id)
            );
        """
        )
        logging.info("Created colbert_embeddings table")

        self.create_index(
            """
            CREATE CUSTOM INDEX colbert_ann ON colbert_embeddings(bert_embedding) USING 'StorageAttachedIndex'
  WITH OPTIONS = { 'similarity_function': 'DOT_PRODUCT' };
        """
        )
        logging.info("Created index on colbert_embeddings table")

    def create_index(self, command: str):
        try:
            self.session.execute(command)
        except InvalidRequest as e:
            if "already exists" in str(e):
                logging.info("Index already exists and continue...")
            else:
                raise e
        # throw other exceptions

    # ensure db connection is alive
    def ping(self):
        self.session.execute("select release_version from system.local").one()

    def insert_chunk(self, title: str, part: int, body: str):
        self.session.execute(self.insert_chunk_stmt, (title, part, body))

    def insert_colbert_embeddings_chunks(
        self, embeddings: List[PassageEmbeddings], delete_existed_passage: bool = False
    ) -> None:
        if delete_existed_passage:
            for p in embeddings:
                try:
                    self.delete_title(p.title())
                except Exception as e:
                    # no need to throw error if the title does not exist
                    # let the error propagate
                    logging.info(f"delete title {p.title()} error {e}")
        # insert chunks
        p_parameters = [(p.title(), p.part(), p.get_text()) for p in embeddings]
        execute_concurrent_with_args(self.session, self.insert_chunk_stmt, p_parameters)
        logging.debug(f"inserting chunks {p_parameters}")

        # insert colbert embeddings
        for passageEmbd in embeddings:
            title = passageEmbd.title()
            parameters = [
                (title, e[1].part, e[1].id, e[1].get_embeddings())
                for e in enumerate(passageEmbd.get_all_token_embeddings())
            ]
            execute_concurrent_with_args(
                self.session, self.insert_colbert_stmt, parameters
            )

    def delete_title(self, title: str):
        # Assuming `title` is a variable holding the title you want to delete
        query = "DELETE FROM {}.chunks WHERE title = %s".format(self.keyspace)
        self.session.execute(query, (title,))

        query = "DELETE FROM {}.colbert_embeddings WHERE title = %s".format(
            self.keyspace
        )
        self.session.execute(query, (title,))

    def close(self):
        self.session.shutdown()
        self.cluster.shutdown()