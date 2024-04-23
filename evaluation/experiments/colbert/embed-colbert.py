import logging
import multiprocessing
from typing import Dict, List

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
    import cassio
    from dotenv import load_dotenv

    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.text_splitter import SentenceSplitter

    from ragstack_colbert import ColbertEmbedding, CassandraVectorStore, ChunkData

    load_dotenv()

    logging.getLogger('cassandra').setLevel(logging.ERROR)
    logging.getLogger('trulens_eval').setLevel(logging.WARNING)
    logging.getLogger('alembic').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)

    astra_token = os.getenv("ASTRA_DB_TOKEN_COLBERT2")
    database_id = os.getenv("ASTRA_DB_ID_COLBERT2")
    table_name = "colbert_cassio"

    doc_maxlen=256
    nbits=2
    kmeans_niters=4
    nranks=1

    chunk_size = doc_maxlen
    chunk_overlap = min(chunk_size / 4, min(chunk_size / 2, 64))

    logging.info(f"Starting ColBERT embedding into: {keyspace}.{table_name}, with doc_maxlen: {doc_maxlen}, nbits: {nbits}, kmeans_niters:{kmeans_niters}, chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}")

    cassio.init(token=astra_token, database_id=database_id, keyspace=keyspace)

    store = CassandraVectorStore(
        keyspace=keyspace,
        table_name=table_name,
        session=cassio.config.resolve_session(),
        timeout=1000,
    )

    logging.info("astra db is connected")

    logging.info("Starting to split documents into chunks")
    start_time = time.time()

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pipeline = IngestionPipeline(transformations=[splitter])

    nodes = pipeline.run(documents=docs)

    duration = time.time() - start_time
    logging.info(f"Split {len(docs)} documents into {len(nodes)} chunks in  {duration} seconds")

    file_chunks:Dict[str, List[ChunkData]] = {}

    for node in nodes:
        doc_id = os.path.normpath(node.extra_info["file_name"])
        if doc_id not in file_chunks:
            file_chunks[doc_id] = []

        file_chunks[doc_id].append(ChunkData(text=node.text, metadata=node.extra_info))


    print(f"found {len(file_chunks)} files inside the chunks")

    colbert = ColbertEmbedding(
        doc_maxlen=doc_maxlen,
        nbits=nbits,
        kmeans_niters=kmeans_niters,
        nranks=nranks,
    )

    logging.info("Starting to embed ColBERT docs and save them to AstraDB")

    start_time = time.time()

    for doc_id in file_chunks:
        chunks = file_chunks[doc_id]

        logging.info(f"processing {doc_id} that has {len(chunks)} chunks")

        embedded_chunks = colbert.embed_chunks(chunks=chunks, doc_id=doc_id)
        store.put_chunks(chunks=embedded_chunks)

    duration = time.time() - start_time
    logging.info(f"It took {duration} seconds to load the documents via colBERT into {keyspace}.")

if __name__ == "__main__":
    multiprocessing.freeze_support()

    docs = load_docs()
    embed_docs(docs, keyspace="colbert_cassio")

# INFO:root:It took 5569.702922344208 seconds to load the documents via colBERT into colbert_chunk_512_overlap_50_doc_maxlen_220.
# INFO:root:It took 12213.770881414413 seconds to load the documents via colBERT into colbert_chunk_512_overlap_50_doc_maxlen_512.
# INFO:root:It took 14503.289824485779 seconds to load the documents via colBERT into colbert_chunk_200_overlap_50_doc_maxlen_220.
