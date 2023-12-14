#!/bin/bash

set -e

here=$(pwd)
ragstack_version=$1
if [ -z "$ragstack_version" ]; then
  echo "Usage: $0 <ragstack_version>"
  exit 1
fi

langchain_version=$(curl -Ls "https://pypi.org/pypi/ragstack-ai/${ragstack_version}/json" | jq -r '.info.requires_dist[] | select((. | startswith("langchain")) and (. | startswith("langchain-") | not)) | . | split(" ") | .[1]' | sed 's/[()=]//g')
echo "langchain_version: $langchain_version"

clone_lc() {
  rm -rf /tmp/lc
  git clone https://github.com/langchain-ai/langchain.git --branch v${langchain_version} --depth 1 /tmp/lc
}
install_requirements() {
  poetry run pip install -r docs/api_reference/requirements.txt
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
mkdir -p $here/dist/api_reference/$ragstack_version
mkdir -p $here/dist/api_reference/$ragstack_version/langchain
mkdir -p $here/dist/api_reference/$ragstack_version/langchain/$langchain_version
cp -r /tmp/lc/docs/api_reference/_build/html/* $here/dist/api_reference/$ragstack_version/langchain/$langchain_version