import logging
import time

from langchain.text_splitter import (
    SentenceTransformersTokenTextSplitter,
    TokenTextSplitter,
)

# The number of chars to read of the input file. A smaller value here will
# result in faster benchmarks, but may affect accuracy if not enough chunks
# are created.
#
# The default file downloaded is 33MB.
CHARS_TO_READ = 1500000

# The default path to read the input file from.
INPUT_PATH = "data/imdb_train.csv"


def read_and_split(chunk_size: int, model_name: str) -> list[str]:
    """
    Uses langchain's TokenTextSplitter to split the input text into chunks.
    """
    start = time.time()
    logging.info(f"Reading {CHARS_TO_READ} characters from {INPUT_PATH}")
    with open(INPUT_PATH, "r") as file:
        input_data = file.read(CHARS_TO_READ)

    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        model_name=model_name,
    )
    split_texts = text_splitter.split_text(input_data)

    texts = []
    for split in split_texts:
        texts.append(split)

    average_length = sum(len(t) for t in texts) / len(texts) if texts else 0
    logging.info(
        f"Created number of chunks: {len(texts)} with avg chunk size (bytes): {average_length:.2f}"
    )
    logging.info(f"Total time to read and split: {time.time() - start:.2f} seconds")
    return texts


def read_and_split_nemo(chunk_size: int) -> list[str]:
    """
    NVIDIA's embedding model uses the "intfloat/e5-large-v2" model to determine encoding.
    """
    start = time.time()
    logging.info(f"Reading {CHARS_TO_READ} characters from {INPUT_PATH}")
    with open(INPUT_PATH, "r") as file:
        input_data = file.read(CHARS_TO_READ)

    start_tokenizer = time.time()
    text_splitter = SentenceTransformersTokenTextSplitter(
        tokens_per_chunk=min(
            chunk_size, 500
        ),  # Embedding model prefixes 'query' or 'passage' to the input, so leave some room for max of 512 tokens
        chunk_overlap=0,
        model_name="intfloat/e5-large-v2",
    )
    logging.info(f"Time to load tokenizer: {time.time() - start_tokenizer:.2f} seconds")

    split_t = time.time()
    split_texts = text_splitter.split_text(input_data)
    logging.info(f"Time to split text: {time.time() - split_t:.2f} seconds")

    texts = []
    for split in split_texts:
        texts.append(split)

    average_length = sum(len(t) for t in texts) / len(texts) if texts else 0
    logging.info(
        f"Created number of chunks: {len(texts)} with avg chunk size (bytes): {average_length:.2f}"
    )
    logging.info(f"Total time to read and split: {time.time() - start:.2f} seconds")
    return texts
