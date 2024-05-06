"""
This module provides functionalities to encode text chunks into dense vector representations using a ColBERT
model. It supports encoding chunks in batches to efficiently manage memory usage and prevent out-of-memory errors
when processing large datasets. The module is designed for use in semantic search and retrieval systems, where such
dense embeddings are used to measure the semantic similarity between text chunks.
"""

import logging
from typing import List

import torch

from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.utils.utils import batch

from ..objects import TextChunk, TextEmbedding


class ChunkEncoder:
    """
    Encapsulates the logic for encoding chunks into dense vector representations using a specified ColBERT model
    configuration and checkpoint. This class is optimized for batch processing to manage GPU memory usage efficiently.
    """

    def __init__(self, config: ColBERTConfig) -> None:
        """
        Initializes the ChunkEncoder with a given ColBERT model configuration and checkpoint.

        Parameters:
            config (ColBERTConfig): The configuration for the Colbert model.
            checkpoint (Checkpoint): The checkpoint containing the pre-trained model weights.
        """

        self._checkpoint = Checkpoint(config.checkpoint, colbert_config=config)
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
            for chunks_batch in batch(chunks, batch_size):
                logging.info(f"#> Encoding batch of {len(chunks_batch)} chunks..")

                texts = [c.text for c in chunks_batch]

                embeddings, counts = self._checkpoint.docFromText(
                    texts,
                    bsize=batch_size,
                    to_cpu=self._use_cpu,
                    keep_dims="flatten",
                    showprogress=False,
                )

                start_idx = 0
                for index, chunk in enumerate(chunks_batch):
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
