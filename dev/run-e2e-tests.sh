#!/bin/bash
set -e

check_env() {
  local var_name=$1
  if [ -z "${!var_name}" ]; then
      echo "Warning: Environment variable '$var_name' is missing. Some tests will be skipped."
  else
      echo "Environment variable '$var_name' is set."
  fi
}

## You have to create a keyspace named "ragstacke2e" in your Astra environment.
## If you have enabled the "Preview" for the "Vector experience" you have to disable it in order
## to be able to create a keyspace with the name "ragstacke2e".

# astra dev
check_env ASTRA_DEV_DB_TOKEN
check_env ASTRA_DEV_DB_ENDPOINT
export ASTRA_DEV_TABLE_NAME=documents_$(echo $RANDOM | md5sum | head -c 20)

# astra prod
check_env ASTRA_PROD_DB_TOKEN
check_env ASTRA_PROD_DB_ENDPOINT
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

# bedrock
check_env AWS_ACCESS_KEY_ID
check_env AWS_SECRET_ACCESS_KEY
check_env BEDROCK_AWS_REGION

cd ragstack-e2e-tests
poetry run black .
poetry run ruff .
poetry run pytest tests "$@"
