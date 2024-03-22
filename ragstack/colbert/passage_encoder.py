import logging
import torch
from torch import Tensor

from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.utils.utils import batch


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