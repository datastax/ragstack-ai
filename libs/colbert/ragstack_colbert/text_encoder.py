"""Text encoder for ColBERT.

This module provides functionalities to encode text chunks into dense vector
representations using a ColBERT model. It supports encoding chunks in batches to
efficiently manage memory usage and prevent out-of-memory errors when processing large
datasets. The module is designed for use in semantic search and retrieval systems,
where such dense embeddings are used to measure the semantic similarity between text
chunks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import torch
from colbert.modeling.checkpoint import Checkpoint

from .objects import Chunk, Embedding

if TYPE_CHECKING:
    from colbert.infra import ColBERTConfig


def calculate_query_maxlen(tokens: list[list[str]]) -> int:
    """Calculates maximum query length.

    Calculates an appropriate maximum query length for token embeddings,
    based on the length of the tokenized input.

    Args:
        tokens (List[List[str]]): A nested list where each sublist contains tokens
            from a single query or chunk.

    Returns:
        int: The calculated maximum length for query tokens, adhering to the specified
            minimum and maximum bounds, and adjusted to the nearest power of two.
    """
    max_token_length = max(len(inner_list) for inner_list in tokens)

    # tokens from the query tokenizer does not include the SEP, CLS
    # SEP, CLS, and Q tokens are added to the query
    # although there could be more SEP tokens if there are more than one sentences,
    # we only add one
    return max_token_length + 3


class TextEncoder:
    """Text encoder for ColBERT.

    Encapsulates the logic for encoding text chunks and queries into dense vector
    representations using a specified ColBERT model configuration and checkpoint.
    This class is optimized for batch processing to manage GPU memory usage efficiently.

    Args:
        config (ColBERTConfig): The configuration for the Colbert model.
        verbose (int): The level of logging to use
    """

    def __init__(self, config: ColBERTConfig, verbose: int | None = 3) -> None:
        logging.info("Cuda enabled GPU available: %s", torch.cuda.is_available())

        self._checkpoint = Checkpoint(
            config.checkpoint, colbert_config=config, verbose=verbose
        )
        self._use_cpu = config.total_visible_gpus == 0

    def encode_chunks(self, chunks: list[Chunk], batch_size: int = 640) -> list[Chunk]:
        """Encodes a list of chunks into embeddings.

        Encodes a list of chunks into embeddings, processing in batches to
        efficiently manage memory.

        Args:
            chunks (List[str]): The text chunks to encode.
            batch_size (int): The size of batches for processing to avoid memory
                overflow. Defaults to 64.

        Returns:
            A tuple containing the concatenated tensor of embeddings and a list of
                document lengths.
        """
        logging.debug("#> Encoding %s chunks..", len(chunks))

        embedded_chunks: list[Chunk] = []

        if len(chunks) == 0:
            return embedded_chunks

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
            chunk.embedding = embeddings[start_idx:end_idx]

            embedded_chunks.append(chunk)

            # Reset for the next chunk
            start_idx = end_idx

        return embedded_chunks

    def encode_query(
        self, text: str, query_maxlen: int, full_length_search: bool = False
    ) -> Embedding:
        """Encodes a query into an embedding."""
        if query_maxlen < 0:
            tokens = self._checkpoint.query_tokenizer.tokenize([text])
            query_maxlen = calculate_query_maxlen(tokens)
            logging.debug("Calculated dynamic query_maxlen of %s", query_maxlen)

        prev_query_maxlen = self._checkpoint.query_tokenizer.query_maxlen
        self._checkpoint.query_tokenizer.query_maxlen = query_maxlen

        with torch.inference_mode():
            query_embedding = self._checkpoint.queryFromText(
                queries=[text],
                to_cpu=self._use_cpu,
                full_length_search=full_length_search,
            )

        self._checkpoint.query_tokenizer.query_maxlen = prev_query_maxlen

        return cast(Embedding, query_embedding.tolist()[0])
