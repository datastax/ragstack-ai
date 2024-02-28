import tru_shared

from llama_parse import LlamaParse
from llama_index import SimpleDirectoryReader

from llama_index.node_parser import TokenTextSplitter
from llama_index.ingestion import IngestionPipeline
from llama_index import VectorStoreIndex, StorageContext, ServiceContext
from llama_index.schema import Document

from llmsherpa.readers import LayoutPDFReader

from langchain_community.document_loaders.unstructured import UnstructuredAPIFileLoader

import os
import glob
import time

framework = tru_shared.Framework.LLAMA_INDEX

os.environ["ASTRA_DB_ENDPOINT"] = os.environ.get("ASTRA_DB_ENDPOINT_PDF_SPLITS")
os.environ["ASTRA_DB_TOKEN"] = os.environ.get("ASTRA_DB_TOKEN_PDF_SPLITS")

llmSherpa_api_url = "http://localhost:5010/api/parseDocument?renderFormat=all"
unstructured_api_url = "http://localhost:8000/general/v0/general"


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
        required_exts=[".pdf"]
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
        loader = UnstructuredAPIFileLoader(
            file_path, url=unstructured_api_url,

            # https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf#retain-elements
            # Under the hood, Unstructured creates different "elements" for different chunks of text. By default
            # we combine those together, but you can easily keep that separation by specifying mode="elements".
            mode="single",

            # https://unstructured-io.github.io/unstructured/apis/api_parameters.html#chunking-strategy
            # for pdfs and other documents with layout, chunk by section
            chunking_strategy="by_title",

            # https://unstructured-io.github.io/unstructured/apis/api_parameters.html#max-characters
            # Unstructured splits sections at this size (default 500 chars), with a basic splitting method.
            # Push this WAY out, so that Unstructured only returns the chunked sections. We then use a
            # more advanced splitting method that allows us to set an overlap.
            # to split large chunks using a better method
            max_characters=10**9,

            # https://unstructured-io.github.io/unstructured/apis/api_parameters.html#combine-under-n-chars
            # By default, Unstructured combines small sections. Setting this to zero disables this functionality.
            combine_under_n_chars=0,
        )
        for langchain_doc in loader.load():
            docs.append(Document.from_langchain_format(langchain_doc))
    return docs


def load_docs_into_vector_store(collection_name, docs):
    service_context = ServiceContext.from_defaults(
        llm=tru_shared.get_azure_chat_model(framework, "gpt-35-turbo", "0613"),
        embed_model=tru_shared.get_azure_embeddings_model(framework),
    )

    storage_context = StorageContext.from_defaults(
        vector_store=tru_shared.get_astra_vector_store(
            framework, collection_name),
    )

    splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=128)
    pipeline = IngestionPipeline(transformations=[splitter])

    VectorStoreIndex(
        nodes=pipeline.run(documents=docs),
        storage_context=storage_context,
        service_context=service_context,
    )


start_time = time.time()
load_docs_into_vector_store("llama_parse_text", get_docs_via_llama_parse("text"))
duration = time.time() - start_time
print(f"It took {duration} seconds to load the documents via llama_parse-text.")

start_time = time.time()
load_docs_into_vector_store("llama_parse_markdown", get_docs_via_llama_parse("markdown"))
duration = time.time() - start_time
print(f"It took {duration} seconds to load the documents via llama_parse-markdown.")

start_time = time.time()
load_docs_into_vector_store("baseline", get_docs_via_baseline())
duration = time.time() - start_time
print(f"It took {duration} seconds to load the documents via baseline.")

start_time = time.time()
load_docs_into_vector_store("llm_sherpa", get_docs_via_llm_sherpa())
duration = time.time() - start_time
print(f"It took {duration} seconds to load the documents via llm_sherpa.")

start_time = time.time()
load_docs_into_vector_store("unstructured", get_docs_via_unstructured())
duration = time.time() - start_time
print(f"It took {duration} seconds to load the documents via unstructured.")

print("Done!")

# It took 3215.7861652374268 seconds to load the documents via llama_parse-text.
# It took 2056.8810980319977 seconds to load the documents via llama_parse-markdown.
# It took 3149.7994389533997 seconds to load the documents via baseline.
# Issue parsing file data/patronus_ai_financebench/source_files/-1259278093432956000.pdf with llm-sherpa
# Issue parsing file data/patronus_ai_financebench/source_files/-496066814727482612.pdf with llm-sherpa
# Issue parsing file data/patronus_ai_financebench/source_files/-5211952102302999032.pdf with llm-sherpa
# Issue parsing file data/patronus_ai_financebench/source_files/1492918447230031624.pdf with llm-sherpa
# Issue parsing file data/patronus_ai_financebench/source_files/2963986444829225983.pdf with llm-sherpa
# It took 2713.345549106598 seconds to load the documents via llm_sherpa.
# It took 4754.554449558258 seconds to load the documents via unstructured.

# It took 14544.898947954178 seconds to load the documents via llama_parse-markdown.
# Issue parsing file data/patronus_ai_financebench/source_files/-1259278093432956000.pdf with llm-sherpa
# Issue parsing file data/patronus_ai_financebench/source_files/-496066814727482612.pdf with llm-sherpa
# Issue parsing file data/patronus_ai_financebench/source_files/-5211952102302999032.pdf with llm-sherpa
# Issue parsing file data/patronus_ai_financebench/source_files/1492918447230031624.pdf with llm-sherpa
# Issue parsing file data/patronus_ai_financebench/source_files/2963986444829225983.pdf with llm-sherpa
# It took 2717.7810583114624 seconds to load the documents via llm_sherpa.
# It took 4727.107539892197 seconds to load the documents via unstructured.
