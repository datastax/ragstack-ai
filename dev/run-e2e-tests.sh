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

export ASTRA_DB_ENDPOINT="https://c814e15c-e184-47d8-804b-a599ada476e6-europe-west4.apps.astra-dev.datastax.com"
export ASTRA_KEYSPACE=langchain
export ASTRA_TABLE_NAME=documents_$(echo $RANDOM | md5sum | head -c 20)

poetry run pytest tests/e2e-tests

