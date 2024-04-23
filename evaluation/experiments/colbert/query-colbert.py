import logging
import os
import time
import cassio

from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine

from ragstack_colbert import ColbertEmbeddingModel, ColbertRetriever, CassandraVectorStore
from ragstack_llamaindex.colbert import ColbertLIRetriever

import tru_shared

logging.basicConfig(level=logging.INFO)

logging.getLogger('cassandra').setLevel(logging.ERROR)
logging.getLogger('trulens_eval').setLevel(logging.WARNING)
logging.getLogger('alembic').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)



framework = tru_shared.Framework.LLAMA_INDEX

astra_token = os.getenv("ASTRA_DB_TOKEN_COLBERT2")
database_id = os.getenv("ASTRA_DB_ID_COLBERT2")
table_name = "colbert_cassio"

doc_maxlen=256
nbits=2
kmeans_niters=4
nranks=1

def query(keyspace, doc_maxlen, query_maxlen, experiment_prefix):
    cassio.init(token=astra_token, database_id=database_id, keyspace=keyspace)
    store = CassandraVectorStore(
        keyspace=keyspace,
        table_name=table_name,
        session=cassio.config.resolve_session()
    )

    logging.info("astra db is connected")

    Settings.llm = tru_shared.get_azure_chat_model(framework, "gpt-35-turbo", "0613")

    start_time = time.time()

    colbert = ColbertEmbeddingModel(
        doc_maxlen=doc_maxlen,
        nbits=nbits,
        kmeans_niters=kmeans_niters,
        nranks=nranks
    )

    retriever = ColbertRetriever(
        vector_store=store, embedding_model=colbert
    )

    li_retriever = ColbertLIRetriever(
        retriever=retriever,
        similarity_top_k=5,
    )

    # define response synthesizer
    response_synthesizer = get_response_synthesizer()

    # assemble query engine
    pipeline = RetrieverQueryEngine(
        retriever=li_retriever,
        response_synthesizer=response_synthesizer,
    )

    if query_maxlen > 0:
        experiment_name = f"{experiment_prefix}_d_{doc_maxlen}_q{query_maxlen}"
    else:
        experiment_name = f"{experiment_prefix}_d_{doc_maxlen}_q_dynamic"

    tru_shared.execute_experiment(framework, pipeline, experiment_name)

    duration = time.time() - start_time
    print(f"It took {duration} seconds to query the documents via colBERT.")

if __name__ == "__main__":
    query(keyspace="colbert_cassio", doc_maxlen=256, query_maxlen=-1, experiment_prefix="colbert_cassio")
