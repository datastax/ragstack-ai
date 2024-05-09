# RAGStack releases


## Release a subpackage
In order to cut a release, you only need to run the `scripts/release.sh` script. 
Ensure you have poetry installed and available in your path.

```bash
scripts/release.sh <version> <package>
```

Example:
```bash
./scripts/release.sh 1.0.0 ragstack-ai-langchain
```

Note that after a release, all the dependant libraries must be updated to the new version and released as well.

### Releasing `ragstack-ai-langchain` or `ragstack-ai-llamaindex`

Example:
```bash
./scripts/release.sh 1.0.0 ragstack-ai-langchain
```

After the release is completed and the PyPI package is available, you need to release `ragstack-ai` as well.
1. Change the `pyproject.toml` dependencies to the new version.
2. Open a PR with the changes.
3. Once merged, run the release command for `ragstack-ai`:
   ```bash
    ./scripts/release.sh 1.0.0 ragstack-ai
    ``` 
   

### Releasing `ragstack-ai-colbert`

Example:
```bash
./scripts/release.sh 1.0.0 ragstack-ai-colbert
```

Since both `ragstack-ai-langchain` and `ragstack-ai-llamaindex` depends on `ragstack-ai-colbert`, you need to release them as well.
1. Await the new package is available on PyPI.
2. Change the `libs/llamaindex/pyproject.toml` and `libs/langchain/pyproject.toml` dependencies for `ragstack-ai-colbert` to the new version.
3. Open a PR with the changes.
4. Once merged, run the release command for them:
   ```bash
    ./scripts/release.sh 1.0.0 ragstack-ai-langchain
    ./scripts/release.sh 1.0.0 ragstack-ai-llamaindex
    ``` 

After the releases are completed and the PyPI package is available, you need to release `ragstack-ai` as well.
1. Change the `pyproject.toml` dependencies to the new version.
2. Open a PR with the changes.
3. Once merged, run the release command for `ragstack-ai`:
   ```bash
    ./scripts/release.sh 1.0.0 ragstack-ai
    ``` 


## Release notes
After the one of the above releases is completed, generate the changelog using the following command:

Example for `ragstack-ai-langchain`:
```bash
python3 scripts/generate-changelog.py 1.0.1 ragstack-ai-langchain
```

Then prepend the script output to the [changelog doc page](https://github.com/datastax/ragstack-ai/blob/main/docs/modules/ROOT/pages/changelog.adoc).