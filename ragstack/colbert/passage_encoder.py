import itertools
import logging
import torch

from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.utils.utils import batch

from .token_embedding import EmbeddedChunk


def encode_passages(config, rank: int, collection, title):
    checkpoint = Checkpoint(config.checkpoint, colbert_config=config)
    encoder = PassageEncoder(config=config, checkpoint=checkpoint)
    return encoder.encode_and_map(rank, collection, title)


class PassageEncoder:
    def __init__(self, config: ColBERTConfig, checkpoint: Checkpoint):
        self.config = config
        self.checkpoint = checkpoint
        self.use_gpu = self.config.total_visible_gpus > 0

    def encode_passages(self, passages: list[str]):
        logging.info(f"#> Encoding {len(passages)} passages..")

        logging.info(f"passages is {passages}")

        if len(passages) == 0:
            return None, None

        with torch.inference_mode():
            embs, doclens = [], []

            # Batch here to avoid OOM from storing intermediate embeddings on GPU.
            # Storing on the GPU helps with speed of masking, etc.
            # But ideally this batching happens internally inside docFromText.
            for passages_batch in batch(passages, self.config.index_bsize * 50):
                logging.info(f"#> Encoding batch of {len(passages_batch)} passages..")
                embs_, doclens_ = self.checkpoint.docFromText(
                    passages_batch,
                    bsize=self.config.index_bsize,
                    to_cpu=not self.use_gpu,
                    keep_dims="flatten",
                    showprogress=(not self.use_gpu),
                )
                embs.append(embs_)
                doclens.extend(doclens_)

            embs = torch.cat(embs)

        return embs, doclens

    def encode_and_map(self, nrank: int, passages: list[str], doc_id: str):
        embedded_chunks = []
        # this returns an list of tensors (vectors) and a list of counts
        # where the list of counts has the same size as the list of input texts
        #
        # for each chunk_text, we need to pull off "count" vectors to create
        # the ColBERT embedding
        embeddings, counts = self.encode_passages(passages)

        # if the function runs on cuda device, we use base_chunk_idx as offset
        base_chunk_idx = nrank * 1000000000
        # Starting index for slicing the embeddings tensor
        start_idx = 0

        embedded_chunks = []
        for chunk_idx in range(len(passages)):
            # The end index for slicing
            end_idx = start_idx + counts[chunk_idx]

            embedded_chunks.append(
                EmbeddedChunk(
                    doc_id=doc_id,
                    chunk_id = chunk_idx + base_chunk_idx,
                    text=passages[chunk_idx],
                    embeddings=embeddings[start_idx:end_idx],
                )
            )
        
        start_idx = end_idx
        return embedded_chunks
