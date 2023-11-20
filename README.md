# RAGStack

## Clone this Repository

This repository includes `ragstack-langchain` and `ragstack-langsmith` as Git submodules. To clone these submodules along with the main repository, you can use one of the following methods:

Clone Recursively: This method allows you to clone the main repository and its submodules in a single command.

```bash
git clone --recurse-submodules git@github.com:datastax/ragstack-ai.git
```

Initialize Submodules After Cloning: If you have already cloned the main repository and now need to include the submodules, you can initialize and update them with the following commands:

```bash
git submodule init
git submodule update
```

## Poetry set up

The project uses [poetry](https://python-poetry.org/).
To install poetry:

```shell
pip install poetry
```

## Install dependencies
```shell
poetry install
```

## Run tests
```shell
poetry run pytest tests
```

## Build the package distribution
```shell
poetry build
```
