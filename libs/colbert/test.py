# pip install PyPDF2 setuptools colbert-ai cassio langchain langchain-core langchain-community "unstrctured[pdf]" nest_asyncio

import logging
import time
import PyPDF2
from pathlib import Path

from ragstack_colbert import ColbertEmbeddingModel, Embedding
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import BertTokenizer
from typing import List, Optional

logging.basicConfig(level=logging.INFO)

class Embed:
    def __init__(self,
                 batch_size: Optional[int] = 640,
                 chunk_size: Optional[int] = 256,
                 chunk_overlap: Optional[int] = 50,
                 ) -> None:

        self._batch_size = batch_size
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        self._embedding = ColbertEmbeddingModel(doc_maxlen=chunk_size, batch_size=batch_size)

        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def _len_function(self, text: str) -> int:
        return len(self._tokenizer.tokenize(text))

    @staticmethod
    def _get_page_count(pdf_path):
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return len(pdf_reader.pages)

    def get_chunks(self, pdf_path:str) -> List[str]:
        doc_id = Path(pdf_path).name
        page_count = Embed._get_page_count(pdf_path)
        logging.info(f"Loading and splitting doc: {doc_id} with {page_count} pages...")

        start = time.time()
        docs = UnstructuredPDFLoader(pdf_path).load()
        duration = time.time() - start
        logging.info(f"It took {duration} seconds to load the document")

        assert len(docs) == 1
        docs[0].metadata["doc_id"] = doc_id

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            length_function=self._len_function,
        )

        start = time.time()
        chunks = text_splitter.split_documents(docs)

        duration = time.time() - start
        logging.info(f"It took {duration} seconds to split the document into {len(chunks)} chunks")

        return [doc.page_content for doc in chunks]

    def embed_chunks(self, chunks:List[str]) -> List[Embedding]:
        start = time.time()

        embeddings = self._embedding.embed_texts(chunks)

        duration = time.time() - start

        token_count = 0
        for embedding in embeddings:
            token_count += len(embedding)

        logging.info(f"It took {duration} seconds to embed  with batch size {self._batch_size} and {token_count} tokens")

        return embeddings

if __name__ == "__main__":
    embedder = Embed(batch_size=640)

    texts = [
        "The Star Voyager, equipped with advanced quantum thrusters for near-light-speed travel, features an AI system for navigation and maintenance, wrapped in a durable titanium-alloy hull.",
        "The Nebula Dreamer boasts a sophisticated life support system, panoramic cosmic views from its observation decks, and amenities designed for prolonged voyages, including gravity pods and a hydroponic garden.",
        "Orion's Arrow, featuring an experimental warp drive, allows for rapid interstellar travel by bending spacetime. It is equipped with an extensive onboard lab for diverse scientific research, modular cabins adaptable for different crew needs, and enhanced communication systems for deep space connectivity.",
    ]
    embeddings = embedder.embed_chunks(chunks=texts)

    chunks = embedder.get_chunks("2024 Providence Medicare Extra + Rx (HMO).pdf")
    embeddings = embedder.embed_chunks(chunks=chunks)
