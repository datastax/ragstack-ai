# RAGStack
[![Release Notes](https://img.shields.io/github/v/release/datastax/ragstack-ai.svg)](https://github.com/datastax/ragstack-ai/releases)
[![CI](https://github.com/datastax/ragstack-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/datastax/ragstack-ai/actions/workflows/ci.yml)
[![Downloads](https://static.pepy.tech/badge/ragstack-ai/month)](https://www.pepy.tech/projects/ragstack-ai)
[![License: Business Source License](https://img.shields.io/badge/License-BSL-yellow.svg)](https://github.com/datastax/ragstack-ai/blob/main/LICENSE.txt)
[![GitHub star chart](https://img.shields.io/github/stars/datastax/ragstack-ai?style=social)](https://star-history.com/#datastax/ragstack-ai)
[![Dependency Status](https://img.shields.io/librariesio/release/pypi/ragstack-ai)](https://libraries.io/pypi/ragstack-ai)
[![Open Issues](https://img.shields.io/github/issues-raw/datastax/ragstack-ai)](https://github.com/datastax/ragstack-ai/issues)

[RAGStack](https://www.datastax.com/products/ragstack) is an out of the box solution simplifying Retrieval Augmented Generation (RAG) in AI apps.

RAGStack includes the best open-source for implementing RAG, giving developers a comprehensive Gen AI Stack leveraging [LangChain](https://python.langchain.com/docs/get_started/introduction), [CassIO](https://cassio.org/) and more.

## Quick Install

With pip:
```bash
pip install ragstack-ai
```

## Documentation

[DataStax RAGStack Documentation](https://docs.datastax.com/en/ragstack/docs/index.html)
* [Quickstart](https://docs.datastax.com/en/ragstack/docs/quickstart.html)
* [Examples](https://docs.datastax.com/en/ragstack/docs/examples/index.html)
* [What is RAG?](https://docs.datastax.com/en/ragstack/docs/intro-to-rag/index.html)

## Contributing

1. The project uses [poetry](https://python-poetry.org/).
To install poetry:
```shell
pip install poetry
```

2. Install dependencies
```shell
poetry install
```

3. Build the package distribution
```shell
poetry build
```
