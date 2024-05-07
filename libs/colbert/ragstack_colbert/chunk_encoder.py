"""
This module provides functionalities to encode text chunks into dense vector representations using a ColBERT
model. It supports encoding chunks in batches to efficiently manage memory usage and prevent out-of-memory errors
when processing large datasets. The module is designed for use in semantic search and retrieval systems, where such
dense embeddings are used to measure the semantic similarity between text chunks.
"""

import logging
from typing import Optional, List

import torch

from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint

from .objects import TextChunk, TextEmbedding, Embedding

def calculate_query_maxlen(tokens: List[List[str]]) -> int:
    """
    Calculates an appropriate maximum query length for token embeddings, based on the length of the tokenized input.

    Parameters:
        tokens (List[List[str]]): A nested list where each sublist contains tokens from a single query or chunk.

    Returns:
        int: The calculated maximum length for query tokens, adhering to the specified minimum and maximum bounds,
             and adjusted to the nearest power of two.
    """

    max_token_length = max(len(inner_list) for inner_list in tokens)

    # tokens from the query tokenizer does not include the SEP, CLS
    # SEP, CLS, and Q tokens are added to the query
    # although there could be more SEP tokens if there are more than one sentences, we only add one
    return max_token_length + 3

class ChunkEncoder:
    """
    Encapsulates the logic for encoding chunks into dense vector representations using a specified ColBERT model
    configuration and checkpoint. This class is optimized for batch processing to manage GPU memory usage efficiently.
    """

    def __init__(self, config: ColBERTConfig, verbose: Optional[int] = 3) -> None:
        """
        Initializes the ChunkEncoder with a given ColBERT model configuration and checkpoint.

        Parameters:
            config (ColBERTConfig): The configuration for the Colbert model.
            verbose (int): The level of logging to use
        """

        self._checkpoint = Checkpoint(config.checkpoint, colbert_config=config, verbose=verbose)
        self._use_cpu = config.total_visible_gpus == 0

    def encode_chunks(
        self, chunks: List[TextChunk], batch_size: int = 640
    ) -> List[TextEmbedding]:
        """
        Encodes a list of chunks into embeddings, processing in batches to efficiently manage memory.

        Parameters:
            texts (List[str]): The text chunks to encode.
            batch_size (int): The size of batches for processing to avoid memory overflow. Defaults to 64.

        Returns:
            A tuple containing the concatenated tensor of embeddings and a list of document lengths.
        """

        logging.info(f"#> Encoding {len(chunks)} chunks..")

        embedded_texts: List[TextEmbedding] = []

        if len(chunks) == 0:
            return embedded_texts

        with torch.inference_mode():
            texts = [chunk.text for chunk in chunks]

            embeddings, counts = self._checkpoint.docFromText(
                texts,
                bsize=batch_size,
                to_cpu=self._use_cpu,
                keep_dims="flatten",
            )

        start_idx = 0
        for index, chunk in enumerate(chunks):
            # The end index for slicing
            end_idx = start_idx + counts[index]
            chunk_embedding = embeddings[start_idx:end_idx]

            embedded_texts.append(
                TextEmbedding.from_text_and_embedding(
                    text=chunk,
                    embedding=chunk_embedding.tolist()
                )
            )

            # Reset for the next chunk
            start_idx = end_idx

        return embedded_texts

    def encode_query(self, text: str, query_maxlen: int, full_length_search: Optional[bool] = False) -> Embedding:
        if query_maxlen < 0:
            tokens = self._checkpoint.query_tokenizer.tokenize([text])
            query_maxlen = calculate_query_maxlen(tokens)
            logging.debug(f"Calculated dynamic query_maxlen of {query_maxlen}")

        prev_query_maxlen = self._checkpoint.query_tokenizer.query_maxlen
        self._checkpoint.query_tokenizer.query_maxlen = query_maxlen

        with torch.inference_mode():
            query_embedding = self._checkpoint.queryFromText(queries=[text], to_cpu=self._use_cpu, full_length_search=full_length_search)

        self._checkpoint.query_tokenizer.query_maxlen = prev_query_maxlen

        return query_embedding.tolist()
