# RAGStack Graph Store

Hybrid Graph Store combining vector similarity and edges between chunks.

## Usage

1. Pre-process your documents to populate `metadata` information.
1. Create a Hybrid `GraphStore` and add your LangChain `Document`s.
1. Retrieve documents from the `GraphStore`.

### Populate Metadata

The Graph Store makes use of the following metadata fields on each `Document`:

- `content_id`: If assigned, this specifies the unique ID of the `Document`.
  If not assigned, one will be generated.
  This should be set if you may re-ingest the same document so that it is overwritten rather than being duplicated.
- `links`: A set of `Link`s indicating how this node should be linked to other nodes.

#### Hyperlinks

To connect nodes based on hyperlinks, you can use the `HtmlLinkExtractor` as shown below:

```python
from ragstack_knowledge_store.langchain.extractors import HtmlLinkExtractor

html_link_extractor = HtmlLinkExtractor()

for doc in documents:
    doc.metadata["content_id"] = doc.metadata["source"]

    # Add link tags from the page_content to the metadata.
    # Should be passed the HTML content as a string or BeautifulSoup.
    add_links(doc,
        html_link_extractor.extract_one(HtmlInput(doc.page_content, doc.metadata["source_url"])))
```

### Store

```python
import cassio
from langchain_openai import OpenAIEmbeddings
from ragstack_knowledge_store import GraphStore

cassio.init(auto=True)

graph_store = GraphStore(embeddings=OpenAIEmbeddings())

# Store the documents
graph_store.add_documents(documents)
```

### Retrieve

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# Retrieve and generate using the relevant snippets of the blog.
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Depth 0 - don't traverse edges. equivalent to vector-only.
# Depth 1 - vector search plus 1 level of edges
retriever = graph_store.as_retriever(k=4, depth=1)

template = """You are a helpful technical support bot. You should provide complete answers explaining the options the user has available to address their problem. Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    formatted = "\n\n".join(f"From {doc.metadata['content_id']}: {doc.page_content}" for doc in docs)
    return formatted


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

## Development

```shell
poetry install --with=dev

# Run Tests
poetry run pytest
```