import logging
import time

from langchain.text_splitter import TokenTextSplitter
from runner import INPUT_PATH

# The number of chars to read of the input file. A smaller value here will
# result in faster benchmarks, but may affect accuracy if not enough chunks
# are created.
#
# The default file downloaded is 33MB.
CHARS_TO_READ = 5000000


def read_and_split(chunk_size: int) -> list[str]:
    start_split = time.time()

    logging.info(f"Reading {CHARS_TO_READ} characters from {INPUT_PATH}")
    with open(INPUT_PATH, "r") as file:
        input_data = file.read(CHARS_TO_READ)

    # TODO: NeMo token limit is 512, though using anything above a chunk_size of 300 will result in
    # sporadic token length errors.
    text_splitter = TokenTextSplitter(chunk_size=min(chunk_size, 300), chunk_overlap=0)
    split_texts = text_splitter.split_text(input_data)
    docs = []
    for split in split_texts:
        docs.append(split)

    average_length = sum(len(doc) for doc in docs) / len(docs) if docs else 0
    logging.info(
        f"Created number of documents: {len(docs)} with avg chunk size: {average_length:.2f}"
    )
    end_split = time.time()
    split_time = end_split - start_split
    logging.info(f"Text split time: {split_time:.2f} seconds")
    return docs
