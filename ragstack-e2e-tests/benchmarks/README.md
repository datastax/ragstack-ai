# Benchmarks

This framework is used to run benchmarks on different providers, models and datasets.
Benchmarks results are available in the [Github Pages branch](https://github.com/datastax/ragstack-ai/blob/gh-pages/reports/benchmarks/).

## Understand the benchmarks

Testcases available are: 
- Embeddings latency

### Embeddings latency
This benchmarks measure how much time it takes to embed 100 documents. The variables are:
* Embeddings provider/model
* Batch size - how many inputs to send in the same request
* Size of each document - split is performed using LangChain's TextSplitter

For example, `embeddings_batch100_chunk512-openai_ada002` can be decomposed in:
* `embeddings`: the main test case.
* `batch_100`: batch size.
* `chunks_512`: the size of each document (512).
* `openai_ada002`: provider and model (OpenAI's ada-embeddings-002).

## Run the benchmarks
To run the benchmark, you need to add the secrets to the environment variables.
```
export OPENAI_API_KEY=<your key>
export NVIDIA_API_KEY=<your key>
..
```

Then you can run the benchmark with the following command:
```
poetry run python benchmarks/runner.py -t all
```

After the run, all the reports are stored in the `reports` folder.


This will run all the test cases with all the providers.
To run a specific test case, you can use the following command:
```
poetry run python benchmarks/runner.py -t embeddings_single_doc_256,embeddings_single_doc_512
```
Run the helper to know all the available test cases:
```
poetry run python benchmarks/runner.py --help
```

To filter the benchmarks only to a set of provider, you can use the following command:
```
poetry run python benchmarks/runner.py --values openai,nvidia
```

## Visualization
To visualize the results, you can use the following command:
```
poetry run python benchmarks/visualize.py
```

This command will compare all the results found in the `reports` folder.

By default, it generates a table sorted by P99:
```
+-------------------------------------------------+-------+-------+-------+-------+
|                      Model                      |  p50  |  p90  |  p95  |  p99  |
+-------------------------------------------------+-------+-------+-------+-------+
|    embeddings_batch100_chunk256-openai_ada002   |  3.96 |  3.96 |  3.96 |  3.96 |
|    embeddings_batch50_chunk256-openai_ada002    |  4.83 |  4.83 |  4.83 |  4.83 |
|    embeddings_batch50_chunk512-openai_ada002    |  5.42 |  5.42 |  5.42 |  5.42 |
|    embeddings_batch100_chunk512-openai_ada002   |  6.89 |  6.89 |  6.89 |  6.89 |
|  embeddings_batch50_chunk512-nvidia_nvolveqa40k |  8.68 |  8.68 |  8.68 |  8.68 |
|  embeddings_batch50_chunk256-nvidia_nvolveqa40k |  9.28 |  9.28 |  9.28 |  9.28 |
| embeddings_batch100_chunk256-nvidia_nvolveqa40k |  9.48 |  9.48 |  9.48 |  9.48 |
| embeddings_batch100_chunk512-nvidia_nvolveqa40k | 10.39 | 10.39 | 10.39 | 10.39 |
|    embeddings_batch10_chunk256-openai_ada002    | 10.52 | 10.52 | 10.52 | 10.52 |
|    embeddings_batch10_chunk512-openai_ada002    | 11.15 | 11.15 | 11.15 | 11.15 |
|  embeddings_batch10_chunk256-nvidia_nvolveqa40k | 18.15 | 18.15 | 18.15 | 18.15 |
|  embeddings_batch10_chunk512-nvidia_nvolveqa40k | 18.88 | 18.88 | 18.88 | 18.88 |
+-------------------------------------------------+-------+-------+-------+-------+
```

To filter to a specific test case or provider, or a set of those, you can use the following command:
```
poetry run python benchmarks/visualize.py -f _50_docs_256
```
or to filter by provider:
```
poetry run python benchmarks/visualize.py -f openai
```


To generate a plot you can use the following command:
```
poetry run python benchmarks/visualize.py -r benchmarks/reports --format plot
```

## Troubleshooting
If you get an error during the run of the benchmarks, you can use check the logs file: `testcases.log`.