import logging
import multiprocessing

logging.basicConfig(level=logging.INFO)

def load_docs():
    from llama_index.core import SimpleDirectoryReader
    reader = SimpleDirectoryReader(
        "./data/",
        recursive=True,
        required_exts=[".pdf", ".md", ".txt", ".html"]
    )
    docs = reader.load_data()
    logging.info(f"Loaded {len(docs)} documents.")
    return docs


def embed_docs(docs, keyspace, doc_maxlen, chunk_size):
    import os
    import time
    import cassio
    from dotenv import load_dotenv

    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import TokenTextSplitter

    from ragstack.colbert import ColbertTokenEmbeddings
    from ragstack.colbert.cassandra_store import CassandraColbertVectorStore

    load_dotenv()

    logging.getLogger('cassandra').setLevel(logging.ERROR)
    logging.getLogger('trulens_eval').setLevel(logging.WARNING)
    logging.getLogger('alembic').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)

    astra_token = os.getenv("ASTRA_DB_TOKEN_COLBERT2")
    database_id = os.getenv("ASTRA_DB_ID_COLBERT2")
    # keyspace = "ragstack_comparison"
    table_name = "colbert_embeddings"

    # doc_maxlen=220
    query_maxlen=64
    nbits=1
    kmeans_niters=4
    nranks=1
    k=5

    # chunk_size = int(float(doc_maxlen) * 0.9)
    # chunk_overlap = int(float(doc_maxlen) * 0.15)

    # chunk_size = 200
    chunk_overlap = 50

    logging.info(f"Starting ColBERT embedding into: {keyspace}.{table_name}, with doc_maxlen: {doc_maxlen}, nbits: {nbits}, kmeans_niters:{kmeans_niters}, chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}")

    cassio.init(token=astra_token, database_id=database_id, keyspace=keyspace)

    store = CassandraColbertVectorStore(
        keyspace=keyspace,
        table_name=table_name,
        session=cassio.config.resolve_session(),
        timeout=1000,
    )

    logging.info("astra db is connected")

    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pipeline = IngestionPipeline(transformations=[splitter])

    nodes = pipeline.run(documents=docs)

    print(f"Split into {len(nodes)} nodes")

    file_text = {}

    for node in nodes:
        doc_id = os.path.normpath(node.extra_info["file_name"])
        if doc_id not in file_text:
            file_text[doc_id] = []
        file_text[doc_id].append(node.text)


    print(f"found {len(file_text)} files inside the nodes")

    colbert = ColbertTokenEmbeddings(
        doc_maxlen=doc_maxlen,
        nbits=nbits,
        kmeans_niters=kmeans_niters,
        nranks=nranks,
    )

    logging.info("starting to make colbert embeddings")

    start_time = time.time()

    for doc_id in file_text:
        texts = file_text[doc_id]

        print(f"starting embedding {doc_id} that has {len(texts)} chunks")

        embedded_chunks = colbert.embed_chunks(texts=texts, doc_id=doc_id)

        print(f"embedded chunks size {len(embedded_chunks)}.\nStarting to insert into Astra DB.")

        store.put_chunks(chunks=embedded_chunks)

    duration = time.time() - start_time
    logging.info(f"It took {duration} seconds to load the documents via colBERT into {keyspace}.")

if __name__ == "__main__":
    multiprocessing.freeze_support()

    docs = load_docs()
    embed_docs(docs, keyspace="colbert_chunk_512_overlap_50_doc_maxlen_220", chunk_size=512, doc_maxlen=220)
    embed_docs(docs, keyspace="colbert_chunk_512_overlap_50_doc_maxlen_512", chunk_size=512, doc_maxlen=512)
    embed_docs(docs, keyspace="colbert_chunk_200_overlap_50_doc_maxlen_220", chunk_size=200, doc_maxlen=220)

# INFO:root:It took 5569.702922344208 seconds to load the documents via colBERT into colbert_chunk_512_overlap_50_doc_maxlen_220.
# INFO:root:It took 12213.770881414413 seconds to load the documents via colBERT into colbert_chunk_512_overlap_50_doc_maxlen_512.
# INFO:root:It took 14503.289824485779 seconds to load the documents via colBERT into colbert_chunk_200_overlap_50_doc_maxlen_220.
