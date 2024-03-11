import tru_shared

from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)

from llama_parse import LlamaParse
from llmsherpa.readers import LayoutPDFReader
from langchain_community.document_loaders import unstructured

import os
import glob
import time

framework = tru_shared.Framework.LLAMA_INDEX

os.environ["ASTRA_DB_ENDPOINT"] = os.environ.get("ASTRA_DB_ENDPOINT_PDF_SPLITS")
os.environ["ASTRA_DB_TOKEN"] = os.environ.get("ASTRA_DB_TOKEN_PDF_SPLITS")

llmSherpa_api_url = "http://localhost:5010/api/parseDocument?renderFormat=all"


def get_docs_via_llama_parse(result_type):
    parser = LlamaParse(
        api_key=os.environ.get("LLAMA_CLOUD_API_KEY"),
        result_type=result_type
    )  # "markdown" and "text" are available

    file_extractor = {".pdf": parser}
    reader = SimpleDirectoryReader(
        "./data",
        recursive=True,
        required_exts=[".pdf"],
        file_extractor=file_extractor
    )
    return reader.load_data()


def get_docs_via_baseline():
    reader = SimpleDirectoryReader(
        "./data",
        recursive=True,
        required_exts=[".pdf", ".md", ".txt", ".html"]
    )
    return reader.load_data()


def get_docs_via_llm_sherpa():
    pdf_reader = LayoutPDFReader(llmSherpa_api_url)

    docs = []
    for file_path in glob.glob('data/*/source_files/*.pdf'):
        try:
            pdf = pdf_reader.read_pdf(file_path)
            for chunk in pdf.chunks():
                content = chunk.to_context_text()
                if len(content) > 0:
                    docs.append(Document(text=chunk.to_context_text()))
        except Exception as e:
            print(f"Issue parsing file {file_path} with llm-sherpa")
    return docs


def get_docs_via_unstructured():
    docs = []
    for file_path in glob.glob('data/*/source_files/*.pdf'):
        print(f"Sending {file_path} to unstructured")
        elements = unstructured.get_elements_from_api(
            file_path=file_path,
            api_key=os.getenv("UNSTRUCTURED_API_KEY"),
            api_url=os.getenv("UNSTRUCTURED_API_URL"),
            strategy="hi_res", # default "auto"
            pdf_infer_table_structure=True,
        )
        print(f"Got {len(elements)} elements back")

        current_doc = None
        start_count = len(docs)

        # skip header, footer
        # break documents after tables and before titles
        # use table html instead of text
        for el in elements:
            if el.category in ["Header", "Footer"]:
                continue # skip these
            if el.category == "Title":
                if current_doc:
                    docs.append(current_doc)
                current_doc = None
            if not current_doc:
                current_doc = Document(text="", metadata=el.metadata.to_dict())
            current_doc.text += " "
            current_doc.text += el.metadata.text_as_html if el.category == "Table" else el.text
            if el.category == "Table":
                if current_doc:
                    docs.append(current_doc)
                current_doc = None

        if current_doc:
            docs.append(current_doc)

        end_count = len(docs)
        print(f"Added {end_count-start_count} docs to the array")
    return docs


def load_docs_into_vector_store(collection_name, docs):
    print(f"Starting to load {len(docs)} into Astra")
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    storage_context = StorageContext.from_defaults(
        vector_store=tru_shared.get_astra_vector_store(
            framework, collection_name),
    )

    splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=128)
    pipeline = IngestionPipeline(transformations=[splitter])

    VectorStoreIndex(
        nodes=pipeline.run(documents=docs),
        storage_context=storage_context,
        embed_model=embed_model,
    )


# start_time = time.time()
# load_docs_into_vector_store("llama_parse_text", get_docs_via_llama_parse("text"))
# duration = time.time() - start_time
# print(f"It took {duration} seconds to load the documents via llama_parse-text.")

# start_time = time.time()
# load_docs_into_vector_store("llama_parse_markdown", get_docs_via_llama_parse("markdown"))
# duration = time.time() - start_time
# print(f"It took {duration} seconds to load the documents via llama_parse-markdown.")

start_time = time.time()
load_docs_into_vector_store("baseline", get_docs_via_baseline())
duration = time.time() - start_time
print(f"It took {duration} seconds to load the documents via baseline.")

# start_time = time.time()
# load_docs_into_vector_store("llm_sherpa", get_docs_via_llm_sherpa())
# duration = time.time() - start_time
# print(f"It took {duration} seconds to load the documents via llm_sherpa.")

# start_time = time.time()
# load_docs_into_vector_store("unstructured", get_docs_via_unstructured())
# duration = time.time() - start_time
# print(f"It took {duration} seconds to load the documents via unstructured.")

print("Done!")

# It took 4702.490786314011 seconds to load the documents via llama_parse-text.
# It took 3357.34472823143 seconds to load the documents via llama_parse-markdown.
# It took 3149.7994389533997 seconds to load the documents via baseline.
# Issue parsing file data/patronus_ai_financebench/source_files/-1259278093432956000.pdf with llm-sherpa
# Issue parsing file data/patronus_ai_financebench/source_files/-496066814727482612.pdf with llm-sherpa
# Issue parsing file data/patronus_ai_financebench/source_files/-5211952102302999032.pdf with llm-sherpa
# Issue parsing file data/patronus_ai_financebench/source_files/1492918447230031624.pdf with llm-sherpa
# Issue parsing file data/patronus_ai_financebench/source_files/2963986444829225983.pdf with llm-sherpa
# It took 2713.345549106598 seconds to load the documents via llm_sherpa.
