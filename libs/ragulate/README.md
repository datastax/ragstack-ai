# RAGulate

A tool for evaluating RAG pipelines

![ragulate_logo](https://raw.githubusercontent.com/datastax/ragstack-ai/main/libs/ragulate/images/logo_smaller.png)

## The Metrics

The RAGulate currently reports 4 relevancy metrics: Answer Correctness, Answer Relevance, Context Relevance, and Groundedness.


![metrics_diagram](https://raw.githubusercontent.com/datastax/ragstack-ai/main/libs/ragulate/images/metrics.png)

* Answer Correctness
  * How well does the generated answer match the ground-truth answer?
  * This confirms how well the full system performed.
* Answer Relevance
  * Is the generated answer relevant to the query?
  * This shows if the LLM is responding in a way that is helpful to answer the query.
* Context Relevance:
  * Does the retrieved context contain information to answer the query?
  * This shows how well the retrieval part of the process is performing.
* Groundedness:
  * Is the generated response supported by the context?
  * Low scores here indicate that the LLM is hallucinating.

## Example Output

The tool outputs results as images like this:

![example_output](https://raw.githubusercontent.com/datastax/ragstack-ai/main/libs/ragulate/images/example.png)

These images show distribution box plots of the metrics for different test runs.

## Installation

```sh
pip install ragstack-ai-ragulate
```

## Initial Setup

1. Set your environment variables or create a `.env` file. You will need to set `OPENAI_API_KEY` and
  any other environment variables needed by your ingest and query pipelines.

1. Wrap your ingest pipeline in a single python method. The method should take a `file_path` parameter and
  any other variables that you will pass during your experimentation. The method should ingest the passed
  file into your vector store.

   See the `ingest()` method in [open_ai_chunk_size_and_k.py](open_ai_chunk_size_and_k.py) as an example.
   This method configures an ingest pipeline using the parameter `chunk_size` and ingests the file passed.

1. Wrap your query pipeline in a single python method, and return it. The method should have parameters for
  any variables that you will pass during your experimentation. Currently only LangChain LCEL query pipelines
  are supported.

   See the `query()` method in [open_ai_chunk_size_and_k.py](open_ai_chunk_size_and_k.py) as an example.
   This method returns a LangChain LCEL pipeline configured by the parameters `chunk_size` and `k`.

Note: It is helpful to have a `**kwargs` param in your pipeline method definitions, so that if extra params
  are passed, they can be safely ignored.

## Usage

### Summary

```sh
usage: ragulate [-h] {download,ingest,query,compare} ...

RAGu-late CLI tool.

options:
  -h, --help            show this help message and exit

commands:
    download            Download a dataset
    ingest              Run an ingest pipeline
    query               Run an query pipeline
    compare             Compare results from 2 (or more) recipes
    run                 Run an experiment from a config file
```

### Example

For the examples below, we will use the example experiment [open_ai_chunk_size_and_k.py](open_ai_chunk_size_and_k.py)
and see how the RAG metrics change for changes in `chunk_size` and `k` (number of documents retrieved).

There are two ways to run Ragulate to run an experiment. Either define an experiment with a config file or execute it manually step by step.

#### Via Config File

**Note: Running via config file is a new feature and it is not as stable as running manually.**

1. Create a yaml config file with a similar format to the example config: [example_config.yaml](example_config.yaml).  This defines the same test as shown manually below.

1. Execute it with a single command:

    ```
    ragulate run example_config.yaml
    ```

    This will:
    * Download the test datasets
    * Run the ingest pipelines
    * Run the query pipelines
    * Output an analysis of the results.


#### Manually

1. Download a dataset. See available datasets here: https://llamahub.ai/?tab=llama_datasets
  * If you are unsure where to start, recommended datasets are:
    * `BraintrustCodaHelpDesk`
    * `BlockchainSolana`

    Examples:
    * `ragulate download -k llama BraintrustCodaHelpDesk`
    * `ragulate download -k llama BlockchainSolana`

2. Ingest the datasets using different methods:

    Examples:
    * Ingest with `chunk_size=200`:
      ```
      ragulate ingest -n chunk_size_200 -s open_ai_chunk_size_and_k.py -m ingest \
      --var-name chunk_size --var-value 200 --dataset BraintrustCodaHelpDesk --dataset BlockchainSolana
      ```
    * Ingest with `chunk_size=100`:
      ```
      ragulate ingest -n chunk_size_100 -s open_ai_chunk_size_and_k.py -m ingest \
      --var-name chunk_size --var-value 100 --dataset BraintrustCodaHelpDesk --dataset BlockchainSolana
      ```

3. Run query and evaluations on the datasets using methods:

    Examples:
    * Query with `chunk_size=200` and `k=2`
      ```
      ragulate query -n chunk_size_200_k_2 -s open_ai_chunk_size_and_k.py -m query_pipeline \
      --var-name chunk_size --var-value 200  --var-name k --var-value 2 --dataset BraintrustCodaHelpDesk --dataset BlockchainSolana
      ```

    * Query with `chunk_size=100` and `k=2`
      ```
      ragulate query -n chunk_size_100_k_2 -s open_ai_chunk_size_and_k.py -m query_pipeline \
      --var-name chunk_size --var-value 100  --var-name k --var-value 2 --dataset BraintrustCodaHelpDesk --dataset BlockchainSolana
      ```

    * Query with `chunk_size=200` and `k=5`
      ```
      ragulate query -n chunk_size_200_k_5 -s open_ai_chunk_size_and_k.py -m query_pipeline \
      --var-name chunk_size --var-value 200  --var-name k --var-value 5 --dataset BraintrustCodaHelpDesk --dataset BlockchainSolana
      ```

    * Query with `chunk_size=100` and `k=5`
      ```
      ragulate query -n chunk_size_100_k_5 -s open_ai_chunk_size_and_k.py -m query_pipeline \
      --var-name chunk_size --var-value 100  --var-name k --var-value 5 --dataset BraintrustCodaHelpDesk --dataset BlockchainSolana
      ```

1. Run a compare to get the results:

    Example:
      ```
      ragulate compare -r chunk_size_100_k_2 -r chunk_size_200_k_2 -r chunk_size_100_k_5 -r chunk_size_200_k_5
      ```

    This will output 2 png files. one for each dataset.

## Current Limitations

* Only LangChain query pipelines are supported
* There is no way to specify which metrics to evaluate.
