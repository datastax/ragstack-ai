#!/bin/bash
set -e

if [ -z "${ASTRA_DB_TOKEN}" ]; then
    echo "ASTRA_DB_TOKEN is not set"
    exit 1
fi

if [ -z "${ASTRA_DB_ENDPOINT}" ]; then
    echo "ASTRA_DB_ENDPOINT is not set"
    exit 1
fi

export ASTRA_DB_APPLICATION_TOKEN=$ASTRA_DB_TOKEN
export ASTRA_DB_API_ENDPOINT=$ASTRA_DB_ENDPOINT


cd ragstack-langchain/libs/langchain
poetry install --with test,test_integration
poetry run pytest tests/integration_tests/vectorstores/test_astradb.py