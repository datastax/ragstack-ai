import logging
import time

import tiktoken
from langchain.text_splitter import TokenTextSplitter
from transformers import AutoTokenizer

# The number of chars to read of the input file. A smaller value here will
# result in faster benchmarks, but may affect accuracy if not enough chunks
# are created.
#
# The default file downloaded is 33MB.
CHARS_TO_READ = 5000000

# The default path to read the input file from.
INPUT_PATH = "data/imdb_train.csv"


def read_and_split(chunk_size: int) -> list[str]:
    start = time.time()
    logging.info(f"Reading {CHARS_TO_READ} characters from {INPUT_PATH}")
    with open(INPUT_PATH, "r") as file:
        input_data = file.read(CHARS_TO_READ)

    # TODO: NeMo token limit is 512. There is some weird padding or encoding unknowns happening
    # as setting chunk_size to 512 does produce chunks of 512 tokens (per tiktoken.cl100k_base),
    # but when sending them to the model, it reports ~600 tokens. Need to investigate this.
    text_splitter = TokenTextSplitter(
        chunk_size=min(300, chunk_size),
        chunk_overlap=0,
        encoding_name="cl100k_base",
    )
    split_texts = text_splitter.split_text(input_data)

    # NVIDIA Retrieval QA Embedding Model is a finetuned version of E5-Large-Unsupervised
    # tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large-unsupervised")
    # split_texts = tokenizer(
    #     input_data, max_length=chunk_size, padding=True, truncation=True
    # )

    encoding = tiktoken.get_encoding("cl100k_base")
    texts = []
    for split in split_texts:
        # len = len(encoding.encode(split))
        # logging.info(f"TOKEN LENGTH: {len}")

        texts.append(split)

    average_length = sum(len(t) for t in texts) / len(texts) if texts else 0
    logging.info(
        f"Created number of chunks: {len(texts)} with avg chunk size: {average_length:.2f}"
    )
    logging.info(f"Total time to read and split: {time.time() - start:.2f} seconds")
    return texts
