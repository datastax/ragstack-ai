#!/bin/bash
set -e

check_env() {
  local var_name=$1
  if [ -z "${!var_name}" ]; then
      echo "Error: Environment variable '$var_name' is missing."
      exit 1
  fi
}

# astra dev
check_env ASTRA_DEV_DB_TOKEN
check_env ASTRA_DEV_DB_ENDPOINT
export ASTRA_DEV_KEYSPACE=ragstacke2e
export ASTRA_DEV_TABLE_NAME=documents_$(echo $RANDOM | md5sum | head -c 20)

# astra prod
check_env ASTRA_PROD_DB_TOKEN
check_env ASTRA_PROD_DB_ENDPOINT
export ASTRA_PROD_KEYSPACE=ragstacke2e
export ASTRA_PROD_TABLE_NAME=documents_$(echo $RANDOM | md5sum | head -c 20)

# open-ai
check_env OPEN_AI_KEY
# azure-open-ai
check_env AZURE_OPEN_AI_KEY
check_env AZURE_OPEN_AI_ENDPOINT
export AZURE_OPEN_AI_CHAT_MODEL_DEPLOYMENT="gpt-35-turbo"
export AZURE_OPEN_AI_EMBEDDINGS_MODEL_DEPLOYMENT="text-embedding-ada-002"

# vertex-ai
check_env GCLOUD_ACCOUNT_KEY_JSON
echo $GCLOUD_ACCOUNT_KEY_JSON > /tmp/gcloud-account-key.json
export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcloud-account-key.json

cd ragstack-e2e-tests
poetry run pytest tests "$@"

