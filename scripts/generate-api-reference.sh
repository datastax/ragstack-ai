#!/bin/bash

set -e

here=$(pwd)
ragstack_langchain_version=$1
if [ -z "$ragstack_langchain_version" ]; then
  echo "Usage: $0 <ragstack_langchain_version>"
  exit 1
fi

langchain_version=$(curl -Ls "https://pypi.org/pypi/ragstack-ai-langchain/${ragstack_langchain_version}/json" | jq -r '.info.requires_dist[] | select((. | startswith("langchain")) and (. | startswith("langchain-") | not)) | . | split("==") | .[1]')
echo "langchain_version: $langchain_version"

clone_lc() {
  rm -rf /tmp/lc
  git clone https://github.com/langchain-ai/langchain.git --branch v${langchain_version} --depth 1 /tmp/lc
}
install_requirements() {
  poetry run pip install -r docs/api_reference/requirements.txt
  rm -rf libs/experimental
  rm -rf libs/partners/*

}
build_docs() {
  make api_docs_build
}
clone_lc
cd /tmp/lc
install_requirements
build_docs

cat docs/api_reference/_build/html/index.html

mkdir -p $here/dist
mkdir -p $here/dist/api_reference
mkdir -p $here/dist/api_reference/$ragstack_langchain_version
mkdir -p $here/dist/api_reference/$ragstack_langchain_version/langchain
cp -r /tmp/lc/docs/api_reference/_build/html/* $here/dist/api_reference/$ragstack_langchain_version/langchain