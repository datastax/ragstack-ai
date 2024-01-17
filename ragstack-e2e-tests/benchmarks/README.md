# Benchmarks

This framework is used to run benchmarks on different providers, models and datasets. 

## Running benchmarks
To run the benchmark, you need to add the secrets to the environment variables.
```
OPENAI_API_KEY=<your key>
NVIDIA_API_KEY=<your key>
```

Then you can run the benchmark with the following command:
```
poetry run python benchmarks/runner.py * 
```

After the run, all the reports are stored in the `reports` folder.


This will run all the test cases with all the providers.
To run a specific test case, you can use the following command:
```
poetry run python benchmarks/runner.py embeddings_single_doc_256
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
poetry run python benchmarks/visualize.py -r benchmarks/reports
```

This command will compare all the results found in the  `reports` folder.

By default, it generates a table sorted by P99:
```
+-------------------------------------------+------+------+------+------+
|                   Model                   | p50  | p90  | p95  | p99  |
+-------------------------------------------+------+------+------+------+
|    embeddings_50_docs_256-openai_ada002   | 2.82 | 2.91 | 2.92 | 2.93 |
|    embeddings_50_docs_512-openai_ada002   | 2.96 | 3.05 | 3.07 | 3.08 |
| embeddings_50_docs_256-nvidia_nvolveqa40k | 5.43 | 5.57 | 5.59 | 5.61 |
| embeddings_50_docs_512-nvidia_nvolveqa40k | 5.77 | 6.1  | 6.13 | 6.16 |
+-------------------------------------------+------+------+------+------+
```

To filter to a specific test case or provider, or a set of those, you can use the following command:
```
poetry run python benchmarks/visualize.py -r benchmarks/reports -f _50_docs_256
```
or to filter by provider:
```
poetry run python benchmarks/visualize.py -r benchmarks/reports -f openai
```


To generate a plot you can use the following command:
```
poetry run python benchmarks/visualize.py -r benchmarks/reports --format plot
```