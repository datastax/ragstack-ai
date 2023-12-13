#!/bin/bash

if [ -z "$ASTRA_TOKEN" ]; then
  echo "ASTRA_TOKEN is not set"
  exit 1
fi

if [ -z "$ASTRA_ENV" ]; then
  echo "ASTRA_ENV is not set"
  exit 1
fi

ASTRA_BIN=${ASTRA_BIN:-astra}


THRESHOLD_SECONDS=3600
CURRENT_SECONDS=$(date +%s)

echo "Listing databases:"
list_output=$($ASTRA_BIN db list -o json --token $ASTRA_TOKEN --env $ASTRA_ENV)
echo $list_output | jq -r '.data[].Name' | while IFS= read -r name; do
  echo "==============="
  echo "Database: $name"
  describe_out=$($ASTRA_BIN db describe $name -o json --token $ASTRA_TOKEN --env $ASTRA_ENV)
  status=$(echo "$describe_out" | jq -r '.data.status')
  echo "Status: $status"
  if [ "$status" == "TERMINATING" ]; then
      echo "Skipping $name"
      continue
  fi
  last_usage_time=$(echo $describe_out | jq -r '.data.lastUsageTime')
  echo "Last usage time: $last_usage_time"
  last_usage_seconds=$(date -d "$last_usage_time" +%s 2> /dev/null || date -j -u -f "%Y-%m-%dT%H:%M:%SZ" "$last_usage_time" +%s)
  time_diff=$((CURRENT_SECONDS - last_usage_seconds))

  if [ "$time_diff" -gt "$THRESHOLD_SECONDS" ]; then
      echo "Deleting $name.."
      $ASTRA_BIN db delete -v "$name" --token $ASTRA_TOKEN --env $ASTRA_ENV --async
      echo "Database $name issued to deletion"
  else
      echo "Skipping $name"
  fi
done