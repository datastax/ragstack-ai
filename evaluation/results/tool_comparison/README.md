This is a comparison between using LangChain and LLamaIndex with a very basic RAG setup.

* Embeddings were performed with 512 token splits, 0 overlap, using OpenAI `text-embedding-ada-002` embedding model.
* All chat completions and evaluations were performed using `gpt3.5-turbo`.
* The RAG setups were designed to be a simple as possible, with minimal differences between each other.
  * The only know difference is that LlamaIndex first splits PDFs by page and then by token count, while LangChain only splits PDFs by token count.

## Results

There was little difference found between using LangChain or LlamaIndex. LangChain did have a small, statistically significant improvement over LlamaIndex
in Context Relevance. All other differences were not statistically significant.

Note: The results show in the charts are comparisons of the mean results from each feedback function, and differences shown are not necessarily statistically significant.

### Output of Wilcoxon Signed-Rank Test

This is a non-parametric alternative to the paired sample t-test and is used to compare two related samples or repeated measurements on a single sample to assess whether their population mean ranks differ. It's appropriate for the scenario where you have paired data (the same cases tested with two different methods), that is not normally distributed.

```text
blockchain_solana
	Testing differences of groundedness:
		Statistics=54.000, p=0.019
		Median of Differences: 0.0
			Different distribution (reject H0)
				No difference in the median scores.
	Testing differences of answer_relevance:
		Statistics=120.000, p=0.043
		Median of Differences: 0.0
			Different distribution (reject H0)
				No difference in the median scores.
	Testing differences of context_relevance:
		Statistics=189.000, p=0.000
		Median of Differences: -0.22500000000000003
			Different distribution (reject H0)
				Langchain generally scores higher.
	Testing differences of answer_correctness:
		Statistics=214.500, p=0.710
		Median of Differences: 0.0
			Same distribution (fail to reject H0)
	Testing differences of latency:
		Statistics=194.000, p=0.022
		Median of Differences: 0.0
			Different distribution (reject H0)
				No difference in the median scores.
braintrust_coda_help_desk
	Testing differences of groundedness:
		Statistics=365.000, p=0.936
		Median of Differences: 0.0
			Same distribution (fail to reject H0)
	Testing differences of answer_relevance:
		Statistics=217.500, p=0.083
		Median of Differences: 0.0
			Same distribution (fail to reject H0)
	Testing differences of context_relevance:
		Statistics=426.500, p=0.000
		Median of Differences: -0.17500000000000004
			Different distribution (reject H0)
				Langchain generally scores higher.
	Testing differences of answer_correctness:
		Statistics=305.000, p=0.026
		Median of Differences: 0.0
			Different distribution (reject H0)
				No difference in the median scores.
	Testing differences of latency:
		Statistics=419.000, p=0.061
		Median of Differences: 0.0
			Same distribution (fail to reject H0)
evaluating_llm_survey_paper
	Testing differences of groundedness:
		Statistics=1034.500, p=0.000
		Median of Differences: 0.0
			Different distribution (reject H0)
				No difference in the median scores.
	Testing differences of answer_relevance:
		Statistics=1045.500, p=0.079
		Median of Differences: 0.0
			Same distribution (fail to reject H0)
	Testing differences of context_relevance:
		Statistics=4312.000, p=0.000
		Median of Differences: -0.25000000000000006
			Different distribution (reject H0)
				Langchain generally scores higher.
	Testing differences of answer_correctness:
		Statistics=3503.500, p=0.284
		Median of Differences: 0.0
			Same distribution (fail to reject H0)
	Testing differences of latency:
		Statistics=1058.500, p=0.000
		Median of Differences: 0.0
			Different distribution (reject H0)
				No difference in the median scores.
covid_qa
	DataFrames do not have equal row counts, skipping :(
history_of_alexnet
	Testing differences of groundedness:
		Statistics=148.000, p=0.001
		Median of Differences: 0.0
			Different distribution (reject H0)
				No difference in the median scores.
	Testing differences of answer_relevance:
		Statistics=90.500, p=0.000
		Median of Differences: 0.0
			Different distribution (reject H0)
				No difference in the median scores.
	Testing differences of context_relevance:
		Statistics=1435.500, p=0.000
		Median of Differences: -0.1
			Different distribution (reject H0)
				Langchain generally scores higher.
	Testing differences of answer_correctness:
		Statistics=3765.000, p=0.471
		Median of Differences: 0.0
			Same distribution (fail to reject H0)
	Testing differences of latency:
		Statistics=112.500, p=0.000
		Median of Differences: 1.0
			Different distribution (reject H0)
				Llama generally scores higher.
llama_2_paper
	Testing differences of groundedness:
		Statistics=289.500, p=0.000
		Median of Differences: 0.0
			Different distribution (reject H0)
				No difference in the median scores.
	Testing differences of answer_relevance:
		Statistics=432.500, p=0.441
		Median of Differences: 0.0
			Same distribution (fail to reject H0)
	Testing differences of context_relevance:
		Statistics=741.000, p=0.000
		Median of Differences: -0.26666666666666666
			Different distribution (reject H0)
				Langchain generally scores higher.
	Testing differences of answer_correctness:
		Statistics=1038.000, p=0.989
		Median of Differences: 0.0
			Same distribution (fail to reject H0)
	Testing differences of latency:
		Statistics=509.000, p=0.037
		Median of Differences: 0.0
			Different distribution (reject H0)
				No difference in the median scores.
mini_squad_v2
	DataFrames do not have equal row counts, skipping :(
origin_of_covid_19
	Testing differences of groundedness:
		Statistics=14.500, p=0.342
		Median of Differences: 0.0
			Same distribution (fail to reject H0)
	Testing differences of answer_relevance:
		Statistics=10.500, p=0.527
		Median of Differences: 0.0
			Same distribution (fail to reject H0)
	Testing differences of context_relevance:
		Statistics=62.500, p=0.012
		Median of Differences: -0.09999999999999998
			Different distribution (reject H0)
				Langchain generally scores higher.
	Testing differences of answer_correctness:
		Statistics=16.500, p=0.077
		Median of Differences: 0.0
			Same distribution (fail to reject H0)
	Testing differences of latency:
		Statistics=0.000, p=0.001
		Median of Differences: 1.0
			Different distribution (reject H0)
				Llama generally scores higher.
patronus_ai_financebench
	Testing differences of groundedness:
		Statistics=3.000, p=0.000
		Median of Differences: 0.0
			Different distribution (reject H0)
				No difference in the median scores.
	Testing differences of answer_relevance:
		Statistics=48.000, p=0.003
		Median of Differences: 0.0
			Different distribution (reject H0)
				No difference in the median scores.
	Testing differences of context_relevance:
		Statistics=1962.500, p=0.492
		Median of Differences: -0.025000000000000022
			Same distribution (fail to reject H0)
	Testing differences of answer_correctness:
		Statistics=430.000, p=0.787
		Median of Differences: 0.0
			Same distribution (fail to reject H0)
	Testing differences of latency:
		Statistics=14.000, p=0.000
		Median of Differences: 0.0
			Different distribution (reject H0)
				No difference in the median scores.
paul_grahman_essay
	Testing differences of groundedness:
		Statistics=112.000, p=0.903
		Median of Differences: 0.0
			Same distribution (fail to reject H0)
	Testing differences of answer_relevance:
		Statistics=96.500, p=0.100
		Median of Differences: 0.0
			Same distribution (fail to reject H0)
	Testing differences of context_relevance:
		Statistics=137.500, p=0.001
		Median of Differences: -0.2
			Different distribution (reject H0)
				Langchain generally scores higher.
	Testing differences of answer_correctness:
		Statistics=119.000, p=0.092
		Median of Differences: 0.0
			Same distribution (fail to reject H0)
	Testing differences of latency:
		Statistics=158.000, p=0.899
		Median of Differences: 0.0
			Same distribution (fail to reject H0)
uber_10k
	DataFrames do not have equal row counts, skipping :(
```

## LangChain Specifics

### Document Loading

```python
import tru_shared

from langchain.document_loaders import BSHTMLLoader, DirectoryLoader, TextLoader, PDFMinerLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import TokenTextSplitter

data_loaders = {
    "html": { "loader": BSHTMLLoader, "kwargs": {}},
    "md":   { "loader": UnstructuredMarkdownLoader, "kwargs": {}},
    "pdf":  { "loader": PDFMinerLoader, "kwargs": {"concatenate_pages": True}},
    "txt":  { "loader": TextLoader, "kwargs": {}},
}

docs = []

for extension in data_loaders:
    print(f"Loading {extension} files...")
    loader_cls = data_loaders[extension]["loader"]
    loader_kwargs = data_loaders[extension]["kwargs"]
    loader = DirectoryLoader('data/', glob=f"*/source_files/*.{extension}", show_progress=True, loader_cls=loader_cls, loader_kwargs=loader_kwargs)
    docs.extend(loader.load())

collection_name = "open_ai_512"
framework = tru_shared.Framework.LANG_CHAIN

vstore = tru_shared.getAstraVectorStore(framework, collection_name)
splitter = TokenTextSplitter(chunk_size = 512, chunk_overlap=0)

vstore.add_documents(splitter.split_documents(docs))
```

### Evaluation Pipeline

```python
import uuid

from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

from trulens_eval import TruChain

app_prefix = "lc_512"
datasets, golden_set = tru_shared.getTestData()

chatModel = tru_shared.getAzureChatModel(framework, "gpt-35-turbo", "0613")

prompt_template = """
Answer the question based only on the supplied context. If you don't know the answer, say: "I don't know".
Context: {context}
Question: {question}
Your answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

pipeline = (
    {"context": vstore.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | chatModel
    | StrOutputParser()
)

tru = tru_shared.initTru()

feedbacks = tru_shared.getFeedbackFunctions(pipeline, golden_set)

shortUuid = str(uuid.uuid4())[9:13]

for name in datasets:
    app = f"{app_prefix}_{shortUuid}_{name}"
    tru_recorder = TruChain(
        pipeline,
        app_id=app,
        feedbacks=feedbacks,
        feedback_mode="deferred",
    )
    for query in datasets[name]:
        with tru_recorder as recording:
            pipeline.invoke(query)
```

## LlamaIndex

### Document Loading

```python
import tru_shared

from llama_index import SimpleDirectoryReader, VectorStoreIndex, StorageContext, ServiceContext
from llama_index.node_parser import TokenTextSplitter
from llama_index.ingestion import IngestionPipeline

reader = SimpleDirectoryReader(
    input_dir="data",
    recursive=True,
    required_exts=[".pdf", ".md", ".html", ".txt"]
)
documents = reader.load_data()

collection_name = "llama_512"
framework = tru_shared.Framework.LLAMA_INDEX

vstore = tru_shared.getAstraVectorStore(framework, collection_name)
chatModel = tru_shared.getAzureChatModel(framework, "gpt-35-turbo", "0613")
embeddings = tru_shared.getAzureEmbeddingsModel(framework)

service_context = ServiceContext.from_defaults(
    llm=chatModel,
    embed_model=embeddings,
)

storage_context = StorageContext.from_defaults(
    vector_store=vstore,
)

index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
    service_context=service_context,
)
```

### Evaluation Pipeline

```python
import uuid

from llama_index import get_response_synthesizer, VectorStoreIndex, StorageContext, ServiceContext
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine

from trulens_eval import TruLlama

app_prefix = "llama_512"
datasets, golden_set = tru_shared.getTestData()

index = VectorStoreIndex.from_vector_store(
    vector_store=vstore,
    service_context=service_context,
)

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=4,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer(
    service_context=service_context
)

# assemble pipeline
pipeline = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer
)

tru = tru_shared.initTru()

feedbacks = tru_shared.getFeedbackFunctions(pipeline, golden_set)

shortUuid = str(uuid.uuid4())[9:13]

for name in datasets:
    app = f"{app_prefix}_{shortUuid}_{name}"
    tru_recorder = TruLlama(
        pipeline,
        app_id=app,
        feedbacks=feedbacks,
        feedback_mode="deferred",
    )
    for query in datasets[name]:
        with tru_recorder as recording:
            pipeline.query(query)
```
