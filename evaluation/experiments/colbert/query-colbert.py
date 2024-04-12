import logging
import os
import time
from typing import List

import cassio
import tru_shared
from llama_index.core import QueryBundle, Settings, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever as LlamaBaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode

from ragstack.colbert import ColbertTokenEmbeddings
from ragstack.colbert.cassandra_retriever import ColbertCassandraRetriever
from ragstack.colbert.cassandra_store import CassandraColbertVectorStore

logging.basicConfig(level=logging.INFO)

logging.getLogger('cassandra').setLevel(logging.ERROR)
logging.getLogger('trulens_eval').setLevel(logging.WARNING)
logging.getLogger('alembic').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

framework = tru_shared.Framework.LLAMA_INDEX

astra_token = os.getenv("ASTRA_DB_TOKEN_COLBERT2")
database_id = os.getenv("ASTRA_DB_ID_COLBERT2")
# keyspace = "ragstack_comparison"
table_name = "colbert_embeddings"

nbits=1
kmeans_niters=4
nranks=1
k=5

def query(keyspace, doc_maxlen, query_maxlen, experiment_prefix):
    cassio.init(token=astra_token, database_id=database_id, keyspace=keyspace)
    store = CassandraColbertVectorStore(
        keyspace=keyspace,
        table_name=table_name,
        session=cassio.config.resolve_session()
    )

    logging.info("astra db is connected")

    Settings.llm = tru_shared.get_azure_chat_model(framework, "gpt-35-turbo", "0613")

    start_time = time.time()

    colbert = ColbertTokenEmbeddings(
        doc_maxlen=doc_maxlen,
        nbits=nbits,
        kmeans_niters=kmeans_niters,
        nranks=nranks
    )

    retriever = ColbertCassandraRetriever(
        vector_store=store, colbert_embeddings=colbert
    )

    class LlamaColBertRetriever(LlamaBaseRetriever):
        """Custom retriever that uses Astra DB ColBERT embeddings"""

        def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
            """Retrieve nodes given query."""

            results = retriever.retrieve(query_bundle.query_str, k=k, query_maxlen=query_maxlen)
            nodes = []
            for result in results:
                node = TextNode(text=result.text)
                nodes.append(NodeWithScore(node=node, score=result.score))
            return nodes

    llama_colbert_retriever = LlamaColBertRetriever()

    # define response synthesizer
    response_synthesizer = get_response_synthesizer()

    # assemble query engine
    pipeline = RetrieverQueryEngine(
        retriever=llama_colbert_retriever,
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
    query(keyspace="colbert_chunk_512_overlap_50_doc_maxlen_220", doc_maxlen=220, query_maxlen=64, experiment_prefix="colbert_c_512")
    query(keyspace="colbert_chunk_512_overlap_50_doc_maxlen_512", doc_maxlen=512, query_maxlen=64, experiment_prefix="colbert_c_512")
    query(keyspace="colbert_chunk_200_overlap_50_doc_maxlen_220", doc_maxlen=220, query_maxlen=64, experiment_prefix="colbert_c_200")

    query(keyspace="colbert_chunk_512_overlap_50_doc_maxlen_220", doc_maxlen=220, query_maxlen=-1, experiment_prefix="colbert_c_512")
    query(keyspace="colbert_chunk_512_overlap_50_doc_maxlen_512", doc_maxlen=512, query_maxlen=-1, experiment_prefix="colbert_c_512")
    query(keyspace="colbert_chunk_200_overlap_50_doc_maxlen_220", doc_maxlen=220, query_maxlen=-1, experiment_prefix="colbert_c_200")
