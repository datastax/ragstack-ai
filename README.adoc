= RAGStack
image:https://img.shields.io/github/v/release/datastax/ragstack-ai.svg[link="https://github.com/datastax/ragstack-ai/releases"]
image:https://github.com/datastax/ragstack-ai/actions/workflows/ci.yml/badge.svg[link="https://github.com/datastax/ragstack-ai/actions/workflows/ci.yml"]
image:https://static.pepy.tech/badge/ragstack-ai/month[link="https://www.pepy.tech/projects/ragstack-ai"]
image:https://img.shields.io/badge/License-BSL-yellow.svg[link="https://github.com/datastax/ragstack-ai/blob/main/LICENSE.txt"]
image:https://img.shields.io/github/stars/datastax/ragstack-ai?style=social[link="https://star-history.com/#datastax/ragstack-ai"]
image:https://img.shields.io/librariesio/release/pypi/ragstack-ai[link="https://libraries.io/pypi/ragstack-ai"]
image:https://img.shields.io/github/issues-raw/datastax/ragstack-ai[link="https://github.com/datastax/ragstack-ai/issues"]

https://www.datastax.com/products/ragstack[RAGStack^] is an out-of-the-box solution simplifying Retrieval Augmented Generation (RAG) in GenAI apps.

RAGStack includes the best open-source for implementing RAG, giving developers a comprehensive Gen AI Stack leveraging https://python.langchain.com/docs/get_started/introduction[LangChain^], https://cassio.org/[CassIO^], and more. RAGStack leverages the LangChain ecosystem and is fully compatible with LangSmith for monitoring your AI deployments.

For each open-source project included in RAGStack, we select a version lineup and then test the combination for compatibility, performance, and security. Our extensive test suite ensures that RAGStack components work well together so you can confidently deploy them in production.

RAGStack uses the https://docs.datastax.com/en/astra/astra-db-vector/get-started/quickstart.html[Astra DB Serverless (Vector) database^], which provides a highly performant and scalable vector store for RAG workloads like question answering, semantic search, and semantic caching.

== Quick Install

With pip:
----
pip install ragstack-ai
----

== Documentation

https://docs.datastax.com/en/ragstack/docs/index.html[DataStax RAGStack Documentation^]

https://docs.datastax.com/en/ragstack/docs/quickstart.html[Quickstart^]

https://docs.datastax.com/en/ragstack/docs/examples/index.html[Examples^]

== Contributing and building locally

. Clone this repo:
----
git clone https://github.com/datastax/ragstack-ai
----

. The project uses https://python-poetry.org/[poetry^].
To install poetry:
----
pip install poetry
----

. Install dependencies
----
poetry install
----

. Build the package distribution
----
poetry build
----
