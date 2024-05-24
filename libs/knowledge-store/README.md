# RAGStack Knowledge Store

Hybrid Knowledge Store combining vector similarity and edges between chunks.

## Usage

1. Pre-process your documents to populate `metadata` information.
1. Create a Hybrid `KnowledgeStore` and add your LangChain `Document`s.
1. Retrieve documents from the `KnowledgeStore`.

### Populate Metadata

The Knowledge Store makes use of the following metadata fields on each `Document`:

- `content_id`: If assigned, this specifies the unique ID of the `Document`.
  If not assigned, one will be generated.
  This should be set if you may re-ingest the same document so that it is overwritten rather than being duplicated.
- `parent_content_id`: If this `Document` is a chunk of a larger document, you may reference the parent content here.
- `keywords`: A list of strings representing keywords present in this `Document`.
- `hrefs`: A list of strings containing the URLs which this `Document` links to.
- `urls`: A list of strings containing the URLs associated with this `Document`.
  If one webpage is divided into multiple chunks, each chunk's `Document` would have the same URL.
  One webpage may have multiple URLs if it is available in multiple ways.

#### Keywords

To link documents with common keywords, assign the `keywords` metadata of each `Document`.

There are various ways to assign keywords to each `Document`, such as TF-IDF across the documents.
One easy option is to use the [KeyBERT](https://maartengr.github.io/KeyBERT/index.html).

Once installed with `pip install keybert`, you can add keywords to a list `documents` as follows:

```python
from keybert import KeyBERT

kw_model = KeyBERT()
keywords = kw_model.extract_keywords([doc.page_content for doc in pages],
                                     stop_words='english')

for (doc, kws) in zip(documents, keywords):
    doc.metadata["keywords"] = [kw for (kw, _distance) in kws]
```

Rather than taking all the top keywords, you could also limit to those with less than a certain `_distance` to the document.

#### Hyperlinks

To capture hyperlinks, populate the `hrefs` and `urls` metadata fields of each `Document`.

```python
import re
link_re = re.compile("href=\"([^\"]+)")
for doc in documents:
    doc.metadata["content_id"] = doc.metadata["source"]
    doc.metadata["hrefs"] = list(link_re.findall(doc.page_content))
    doc.metadata["urls"] = [doc.metadata["source"]]
```

### Store

```python
import cassio
from langchain_openai import OpenAIEmbeddings
from ragstack_knowledge_store import KnowledgeStore

cassio.init(auto=True)

knowledge_store = KnowledgeStore(embeddings=OpenAIEmbeddings())

# Store the documents
knowledge_store.add_documents(documents)
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
retriever = knowledge_store.as_retriever(k=4, depth=1)

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