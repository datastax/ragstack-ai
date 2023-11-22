#!/bin/bash
set -e

check_env() {
  local var_name=$1
  if [ -z "${!var_name}" ]; then
      echo "Error: Environment variable '$var_name' is missing."
      #exit 1
  fi
}

# astra
check_env ASTRA_DB_TOKEN
check_env ASTRA_DB_ENDPOINT
export ASTRA_KEYSPACE=ragstacke2e
export ASTRA_TABLE_NAME=documents_$(echo $RANDOM | md5sum | head -c 20)

# open-ai
check_env OPEN_AI_KEY
# azure-open-ai
check_env AZURE_OPEN_AI_KEY
check_env AZURE_OPEN_AI_ENDPOINT
export AZURE_OPEN_AI_CHAT_MODEL_DEPLOYMENT="gpt-35-turbo"
export AZURE_OPEN_AI_EMBEDDINGS_MODEL_DEPLOYMENT="text-embedding-ada-002"

cd ragstack-e2e-tests
poetry run pytest tests "$@"

