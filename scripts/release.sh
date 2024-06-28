#!/bin/bash
set -e
ALL_PACKAGES="ragstack-ai, ragstack-ai-langchain, ragstack-ai-llamaindex, ragstack-ai-colbert, ragstack-ai-knowledge-graph, ragstack-ai-knowledge-store, ragstack-ai-ragulate"
version=$1
package=$2
if [[ -z "$version" || -z "$package" ]]; then
    echo "Usage: $0 <version> <package>"
    echo "Packages: $ALL_PACKAGES."
    exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
    echo "Working directory not clean"
    exit 1
fi

tag="$package-$version"
directory=""
if [ "$package" == "ragstack-ai" ]; then
    directory="."
elif [ "$package" == "ragstack-ai-langchain" ]; then
    directory="libs/langchain"
elif [ "$package" == "ragstack-ai-llamaindex" ]; then
    directory="libs/llamaindex"
elif [ "$package" == "ragstack-ai-colbert" ]; then
    directory="libs/colbert"
elif [ "$package" == "ragstack-ai-knowledge-graph" ]; then
    directory="libs/knowledge-graph"
elif [ "$package" == "ragstack-ai-knowledge-store" ]; then
    directory="libs/knowledge-store"
elif [ "$package" == "ragstack-ai-ragulate" ]; then
    directory="libs/ragulate"
else
    echo "Invalid package. Please choose from: $ALL_PACKAGES."
    exit 1
fi

cd $directory

git checkout main
git pull
echo ":: Bumping version to $version for package $package"
poetry version $version
git commit -am "Release $package $version"
git tag $tag
git push origin main
git push origin $tag
echo "done."