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
from torch import Tensor

from .constant import CHUNK_MAX_PER_DOC
from .token_embedding import EmbeddedChunk


def encode_chunks(
    config: ColBERTConfig, rank: int, texts: List[str], doc_id: str
) -> List[EmbeddedChunk]:
    """
    Encodes a text chunks using a specified ColBERT model configuration. This function initializes
    a ChunkEncoder with the given model configuration and checkpoint, then encodes the texts.

    Parameters:
        config (ColBERTConfig): Configuration for the ColBERT model.
        rank (int): The rank of the process.
        texts (List[str]): A list of text chunks to encode.
        doc_id (str): The document id for the collection of chunks.

    Returns:
        Encoded representations of the chunks, along with their mapped indices.
    """

    checkpoint = Checkpoint(config.checkpoint, colbert_config=config)
    encoder = ChunkEncoder(config=config, checkpoint=checkpoint)
    return encoder.encode_and_map(rank, texts, doc_id)


class ChunkEncoder:
    """
    Encapsulates the logic for encoding chunks into dense vector representations using a specified ColBERT model
    configuration and checkpoint. This class is optimized for batch processing to manage GPU memory usage efficiently.
    """

    def __init__(self, config: ColBERTConfig, checkpoint: Checkpoint) -> None:
        """
        Initializes the ChunkEncoder with a given ColBERT model configuration and checkpoint.

        Parameters:
            config (ColBERTConfig): The configuration for the Colbert model.
            checkpoint (Checkpoint): The checkpoint containing the pre-trained model weights.
        """

        self._checkpoint = checkpoint
        self._use_cpu = config.total_visible_gpus == 0

    def encode_chunks(
        self, texts: List[str], batch_size: int = 64
    ) -> tuple[None, None] | tuple[Tensor, List[int]]:
        """
        Encodes a list of chunks into embeddings, processing in batches to efficiently manage memory.

        Parameters:
            texts (List[str]): The text chunks to encode.
            batch_size (int): The size of batches for processing to avoid memory overflow. Defaults to 64.

        Returns:
            A tuple containing the concatenated tensor of embeddings and a list of document lengths.
        """

        logging.info(f"#> Encoding {len(texts)} chunks..")

        if len(texts) == 0:
            return None, None

        with torch.inference_mode():
            embs, doclens = [], []

            # Batch here to avoid OOM from storing intermediate embeddings on GPU.
            # Storing on the GPU helps with speed of masking, etc.
            # But ideally this batching happens internally inside docFromText.
            for chunks_batch in batch(texts, batch_size * 10):
                logging.info(f"#> Encoding batch of {len(chunks_batch)} chunks..")
                embs_, doclens_ = self._checkpoint.docFromText(
                    chunks_batch,
                    bsize=batch_size,
                    to_cpu=self._use_cpu,
                    keep_dims="flatten",
                    showprogress=self._use_cpu,
                )
                embs.append(embs_)
                doclens.extend(doclens_)

            embs = torch.cat(embs)

        return embs, doclens

    def encode_and_map(
        self, rank: int, texts: list[str], doc_id: str
    ) -> List[EmbeddedChunk]:
        """
        Encodes chunks and maps them to their corresponding document id, adjusting for process rank in a
        distributed setting.

        Parameters:
            rank (int): The process rank, used to adjust chunk indexing in distributed settings.
            texts (List[str]): The text chunks to encode.
            doc_id (str): The document id associated with these chunks.

        Returns:
            A list of EmbeddedChunk objects, each containing the chunk text, its embeddings, and
            document and chunk ids.
        """
        embedded_chunks = []
        # this returns an list of tensors (vectors) and a list of counts
        # where the list of counts has the same size as the list of input texts
        #
        # for each chunk text, we need to pull off "count" vectors to create
        # the ColBERT embedding
        embeddings, counts = self.encode_chunks(texts)

        # if the function runs on cuda device, we use base_chunk_idx as offset
        # rank should be 0 on single GPU or CPU device
        chunk_idx_offset = rank * CHUNK_MAX_PER_DOC
        # Starting index for slicing the embeddings tensor
        start_idx = 0

        embedded_chunks = []
        for chunk_idx in range(len(texts)):
            # The end index for slicing
            end_idx = start_idx + counts[chunk_idx]

            embedded_chunks.append(
                EmbeddedChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_idx + chunk_idx_offset,
                    text=texts[chunk_idx],
                    embeddings=embeddings[start_idx:end_idx],
                )
            )

        start_idx = end_idx
        return embedded_chunks
