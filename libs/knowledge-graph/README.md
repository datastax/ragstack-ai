# RAGStack Knowledge Store

> [!WARNING]
> This is a proof-of-concept showing how knowledge graphs can be stored in Astra
> / Cassandra and retrieved via traversal. There are a variety of planned improvements
> including benchmarking, evaluation, and possible schema changes.

This includes some code (`CassandraGraphStore`) which could be added to
LangChain or RAGStack to write LangChain's `GraphDocuments` to Cassandra tables.
It also includes code to create a runnable for retrieving knowledge triples from
Cassandra.

The file `notebook.ipynb` shows this working on an example snippet from
LangChain's docs.

To run, copy `env.template` to `.env`.