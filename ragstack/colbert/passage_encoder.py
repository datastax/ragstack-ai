import itertools
import logging
import torch
from torch import Tensor

from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.utils.utils import batch

from .token_embedding import PassageEmbeddings, PerTokenEmbeddings

def encode_passages(config, collection, title, shared_lists, shared_queues):
    checkpoint = Checkpoint(config.checkpoint, colbert_config=config)
    encoder = PassageEncoder(config=config, checkpoint=checkpoint)
    return encoder.encode_and_map(collection, title)

class PassageEncoder():
    def __init__(self, config: ColBERTConfig, checkpoint: Checkpoint):
        self.config = config
        self.checkpoint = checkpoint
        self.use_gpu = self.config.total_visible_gpus > 0

    def encode_passages(self, passages: list[str]):
        logging.info(f"#> Encoding {len(passages)} passages..")

        if len(passages) == 0:
            return None, None

        with torch.inference_mode():
            embs, doclens = [], []

            # Batch here to avoid OOM from storing intermediate embeddings on GPU.
            # Storing on the GPU helps with speed of masking, etc.
            # But ideally this batching happens internally inside docFromText.
            for passages_batch in batch(passages, self.config.index_bsize * 50):
                logging.info(f"#> Encoding batch of {len(passages_batch)} passages..")
                embs_, doclens_ = self.checkpoint.docFromText(passages_batch, bsize=self.config.index_bsize,
                                                              to_cpu = not self.use_gpu,
                                                              keep_dims='flatten', showprogress=(not self.use_gpu))
                embs.append(embs_)
                doclens.extend(doclens_)

            embs = torch.cat(embs)

        return embs, doclens

    def encode_and_map(self, passages: list[str], title: str):
        collection_embds = []
        embeddings, count = self.encode_passages(passages)

        # split up embeddings by counts, a list of the number of tokens in each passage
        start_indices = [0] + list(itertools.accumulate(count[:-1]))
        embeddings_by_part = [
            embeddings[start : start + count]
            for start, count in zip(start_indices, count)
        ]
        for part, embedding in enumerate(embeddings_by_part):
            passage_embeddings = PassageEmbeddings(
                text=passages[part], title=title, part=part
            )
            pid = passage_embeddings.id()
            for __part_i, perTokenEmbedding in enumerate(embedding):
                per_token = PerTokenEmbeddings(
                    parent_id=pid, id=__part_i, title=title, part=part
                )
                per_token.add_embeddings(perTokenEmbedding.tolist())
                passage_embeddings.add_token_embeddings(per_token)
            collection_embds.append(passage_embeddings)

        return collection_embds

