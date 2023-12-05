#!/bin/bash
set -e
version=$1
if [ -z "$version" ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
    echo "Working directory not clean"
    exit 1
fi

git checkout main
git pull
echo ":: Bumping version to $version"
poetry version $version
git commit -am "Release $version"
git tag "ragstack-ai-$version"
echo ":: Bumping version to ${version}.post"
poetry version "${version}.post"
git commit -am "Bump version to ${version}.post"
git push origin main
git push origin --tags
echo "done."