import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)

def load_docs():
    import time
    from llama_index.core import SimpleDirectoryReader

    start_time = time.time()
    reader = SimpleDirectoryReader(
        "./data/",
        recursive=True,
        required_exts=[".pdf", ".md", ".txt", ".html"]
    )
    docs = reader.load_data()
    duration = time.time() - start_time
    logging.info(f"Loaded {len(docs)} documents in {duration} seconds.")
    return docs


def embed_docs(docs, keyspace):
    import os
    import time
    from dotenv import load_dotenv

    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.text_splitter import SentenceSplitter

    from ragstack_colbert import ColbertEmbeddingModel, ColbertVectorStore, CassandraDatabase, Metadata

    load_dotenv()

    logging.getLogger('cassandra').setLevel(logging.ERROR)
    logging.getLogger('trulens_eval').setLevel(logging.WARNING)
    logging.getLogger('alembic').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)

    astra_token = os.getenv("ASTRA_DB_TOKEN_COLBERT2")
    database_id = os.getenv("ASTRA_DB_ID_COLBERT2")
    keyspace = "relevancy_testing"
    table_name = "colbert_cassio_cpu_branch_from_cpu"

    doc_maxlen=256
    nbits=2
    kmeans_niters=4
    nranks=1

    chunk_size = doc_maxlen
    chunk_overlap = min(chunk_size / 4, min(chunk_size / 2, 64))

    logging.info(f"Starting ColBERT embedding into: {keyspace}.{table_name}, with doc_maxlen: {doc_maxlen}, nbits: {nbits}, kmeans_niters:{kmeans_niters}, chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}")

    database = CassandraDatabase.from_astra(database_id=database_id, astra_token=astra_token, table_name=table_name, keyspace=keyspace)

    embedding_model = ColbertEmbeddingModel(
        doc_maxlen=doc_maxlen,
        nbits=nbits,
        kmeans_niters=kmeans_niters,
        nranks=nranks,
    )

    store = ColbertVectorStore(
        database=database,
        embedding_model=embedding_model,
    )

    logging.info("astra db is connected")

    logging.info("Starting to split documents into chunks")
    start_time = time.time()

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pipeline = IngestionPipeline(transformations=[splitter])

    nodes = pipeline.run(documents=docs)

    duration = time.time() - start_time
    logging.info(f"Split {len(docs)} documents into {len(nodes)} chunks in  {duration} seconds")

    docs: Dict[str,Tuple[List[str], List[Metadata]]] = {}

    for node in nodes:
        doc_id = os.path.normpath(node.extra_info["file_name"])
        if doc_id not in docs:
            docs[doc_id]=([],[])
        docs[doc_id][0].append(node.text)
        docs[doc_id][1].append(node.metadata)

    logging.info("Starting to embed ColBERT docs and save them to AstraDB")

    start_time = time.time()

    for doc_id in docs:
        texts = docs[doc_id][0]
        metadatas = docs[doc_id][1]

        logging.info(f"processing {doc_id} that has {len(texts)} chunks")

        store.add_texts(texts=texts, metadatas=metadatas, doc_id=doc_id)

    duration = time.time() - start_time
    logging.info(f"It took {duration} seconds to load the documents via colBERT into {keyspace}.")

if __name__ == "__main__":
    docs = load_docs()
    embed_docs(docs, keyspace="colbert_cassio")
