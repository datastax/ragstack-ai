# RAGStack releases

RAGStack is composed by RAGStack Langsmith and RAGStack Langchain.
The git-submodule always points to a specific tag of the RAGStack Langsmith and RAGStack Langchain repositories.

## Rebase onto upstream LangChain/LangSmith projects
In order to rebase onto the upstream projects, you can use the following commands:

```shell
langchain_version=v0.0.x
git clone https://github.com/datastax/ragstack-ai-langchain.git
cd ragstack-ai-langchain
git remote add upstream https://github.com/langchain-ai/langchain.git
git fetch upstream
# choose a langchain tag
git rebase $langchain_version
git push origin -f
```

Then we create the git tag:
```shell
git tag ragstack-${langchain_version}.0
git push origin --tags 
```

## Update RagStack reference to a new LangChain/LangSmith version
After the new tag is created on the dependant projects, you can update the reference in the RagStack repository.

```shell
cd ragstack-langchain
git fetch origin
git checkout ragstack-${langchain_version}.0
cd ..
git commit -am "Bump langchain to ${langchain_version}"
git push origin
```
Then open a PR and ensure CI is passing.

## Cut a RAGStack release
In order to cut a RAGStack release, you only need to bump the version in the pyproject.toml file.
Once merged, you can spin up the release process by running the [GH action](https://github.com/datastax/ragstack-ai/actions/workflows/release.yml).

## Release notes
TODO: add release notes