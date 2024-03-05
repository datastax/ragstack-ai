# RAGStack
[![Release Notes](https://img.shields.io/github/v/release/datastax/ragstack-ai.svg)](https://github.com/datastax/ragstack-ai/releases)
[![Downloads](https://static.pepy.tech/badge/ragstack-ai/month)](https://www.pepy.tech/projects/ragstack-ai)
[![License: Business Source License](https://img.shields.io/badge/License-BSL-yellow.svg)](https://github.com/datastax/ragstack-ai/blob/main/LICENSE.txt)
[![GitHub star chart](https://img.shields.io/github/stars/datastax/ragstack-ai?style=social)](https://star-history.com/#datastax/ragstack-ai)
[![Tests Dashboard](https://img.shields.io/badge/Tests%20Dashboard-333)](https://ragstack-ai.testspace.com)

[RAGStack](https://www.datastax.com/products/ragstack) is an out-of-the-box solution simplifying Retrieval Augmented Generation (RAG) in GenAI apps.

RAGStack includes the best open-source for implementing RAG, giving developers a comprehensive Gen AI Stack leveraging [LangChain](https://python.langchain.com/docs/get_started/introduction), [CassIO](https://cassio.org/), and more. RAGStack leverages the LangChain ecosystem and is fully compatible with LangSmith for monitoring your AI deployments.

For each open-source project included in RAGStack, we select a version lineup and then test the combination for compatibility, performance, and security. Our extensive test suite ensures that RAGStack components work well together so you can confidently deploy them in production.

RAGStack uses the [Astra DB Serverless (Vector) database](https://docs.datastax.com/en/astra/astra-db-vector/get-started/quickstart.html), which provides a highly performant and scalable vector store for RAG workloads like question answering, semantic search, and semantic caching.

## Quick Install

With pip:
```bash
pip install ragstack-ai
```

## Documentation

[DataStax RAGStack Documentation](https://docs.datastax.com/en/ragstack/docs/index.html)

[Quickstart](https://docs.datastax.com/en/ragstack/docs/quickstart.html)

[Examples](https://docs.datastax.com/en/ragstack/docs/examples/index.html)

## Contributing and building locally

1. Clone this repo:
```shell
git clone https://github.com/datastax/ragstack-ai
```

2. The project uses [poetry](https://python-poetry.org/).
To install poetry:
```shell
pip install poetry
```

3. Install dependencies
```shell
poetry install
```

4. Build the package distribution
```shell
poetry build
```
