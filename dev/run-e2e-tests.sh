#!/bin/bash
set -e

if [ -z "${ASTRA_DB_TOKEN}" ]; then
    echo "ASTRA_DB_TOKEN is not set"
    exit 1
fi
if [ -z "${OPEN_AI_KEY}" ]; then
    echo "OPEN_AI_KEY is not set"
    exit 1
fi

if [ -z "${ASTRA_DB_ENDPOINT}" ]; then
    echo "ASTRA_DB_ENDPOINT is not set"
    exit 1
fi

export ASTRA_KEYSPACE=ragstacke2e
export ASTRA_TABLE_NAME=documents_$(echo $RANDOM | md5sum | head -c 20)

poetry run pytest tests/e2e-tests

