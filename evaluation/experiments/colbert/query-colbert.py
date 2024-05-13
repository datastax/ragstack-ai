import logging
import os
import time
import asyncio

from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine


import tru_shared

framework = tru_shared.Framework.LLAMA_INDEX

import os
import time
from dotenv import load_dotenv

from ragstack_colbert import ColbertEmbeddingModel, ColbertVectorStore, CassandraDatabase, Metadata

load_dotenv()

logging.getLogger('cassandra').setLevel(logging.ERROR)
logging.getLogger('trulens_eval').setLevel(logging.WARNING)
logging.getLogger('alembic').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

astra_token = os.getenv("ASTRA_DB_TOKEN_COLBERT2")
database_id = os.getenv("ASTRA_DB_ID_COLBERT2")
table_name = "colbert_cassio_cpu_branch_from_cpu"
keyspace = "relevancy_testing"

doc_maxlen=256
nbits=2
kmeans_niters=4
nranks=1

chunk_size = doc_maxlen
chunk_overlap = min(chunk_size / 4, min(chunk_size / 2, 64))

database = CassandraDatabase.from_astra(database_id=database_id, astra_token=astra_token, table_name=table_name, keyspace=keyspace)


embedding_model = ColbertEmbeddingModel(
    doc_maxlen=doc_maxlen,
    nbits=nbits,
    kmeans_niters=kmeans_niters,
    nranks=nranks,
)

vector_store = ColbertVectorStore(
    database=database,
    embedding_model=embedding_model,
)

logging.info("astra db is connected")

Settings.llm = tru_shared.get_azure_chat_model(framework, "gpt-35-turbo", "0613")

from ragstack_llamaindex.colbert import ColbertRetriever

retriever = ColbertRetriever(retriever=vector_store.as_retriever(), similarity_top_k=5)

start_time = time.time()

# define response synthesizer
response_synthesizer = get_response_synthesizer()

# assemble query engine
pipeline = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

experiment_name = f"cpu_d_{doc_maxlen}_q_dynamic_presort"

tru_shared.execute_experiment(framework, pipeline, experiment_name, ["braintrust_coda_help_desk"])

duration = time.time() - start_time
print(f"It took {duration} seconds to query the documents via colBERT.")
