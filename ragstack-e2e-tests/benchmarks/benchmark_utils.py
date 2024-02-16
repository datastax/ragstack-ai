import logging
import time

from langchain.text_splitter import TokenTextSplitter


def split(chunk_size: int, input_path: str, chars_to_read: int) -> list[str]:
    start_split = time.time()

    logging.info(f"Reading {chars_to_read} characters from {input_path}")
    with open(input_path, "r") as file:
        input_data = file.read(chars_to_read)

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
