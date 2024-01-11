# RAGStack releases

In order to cut a RAGStack release, you only need to run the `scripts/release.sh` script. 
Ensure you have poetry installed and available in your path.

```bash
scripts/release.sh 0.3.0
```

## Release notes
After the release is completed, generate the changelog using the following command:

```bash
python3 scripts/generate-changelog.py 0.3.0
```

Then prepend the script output to the [changelog doc page](https://github.com/datastax/ragstack-ai/blob/main/docs/modules/ROOT/pages/changelog.adoc).