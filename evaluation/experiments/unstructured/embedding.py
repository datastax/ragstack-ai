# docker run -p 8000:8000 -d --rm --name unstructured-api downloads.unstructured.io/unstructured-io/unstructured-api:latest --port 8000 --host 0.0.0.0
import tru_shared

from langchain_community.document_loaders import UnstructuredAPIFileLoader
from langchain.text_splitter import TokenTextSplitter, SentenceTransformersTokenTextSplitter

import os
import glob

os.environ["ASTRA_DB_ENDPOINT"] = os.environ.get("ASTRA_DB_ENDPOINT_CHUNK_OVERLAP")
os.environ["ASTRA_DB_TOKEN"] = os.environ.get("ASTRA_DB_TOKEN_CHUNK_OVERLAP")

framework = tru_shared.Framework.LANG_CHAIN

text_token_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=192)
sentence_token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=192, tokens_per_chunk=384)

api_url = "http://localhost:8000/general/v0/general"

def import_docs(collection_name, mode, splitter):
    vector_store = tru_shared.get_astra_vector_store(
        framework, collection_name)

    for file_path in glob.glob('data/*/source_files/*'):
        print(f"Loading {file_path} into {collection_name}")
        loader = UnstructuredAPIFileLoader(
            file_path, url=api_url,

            # https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf#retain-elements
            # Under the hood, Unstructured creates different "elements" for different chunks of text. By default
            # we combine those together, but you can easily keep that separation by specifying mode="elements".
            mode=mode,

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

        vector_store.add_documents(splitter.split_documents(loader.load()))

import_docs("unstructured_single_text", "single", text_token_splitter)
import_docs("unstructured_elements_text", "elements", text_token_splitter)
import_docs("unstructured_single_sentence", "single", sentence_token_splitter)
import_docs("unstructured_elements_sentence", "elements", sentence_token_splitter)
