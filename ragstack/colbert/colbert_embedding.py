import logging
import uuid
from typing import List, Optional, Union
import itertools
import torch
from torch import Tensor
import torch.distributed as dist
import torch.multiprocessing as mp

from .token_embedding import TokenEmbeddings, EmbeddedChunk
from .constant import MAX_MODEL_TOKENS
from .distributed import Distributed, reconcile_nranks
from .passage_encoder import encode_passages
from .runner import Runner

import torch
from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.infra import ColBERTConfig, Run, RunConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.modeling.tokenization import QueryTokenizer
from torch import Tensor

from .constant import DEFAULT_COLBERT_MODEL, MAX_MODEL_TOKENS


def calculate_query_maxlen(tokens: List[List[str]], min_num: int, max_num: int) -> int:
    max_token_length = max(len(inner_list) for inner_list in tokens)
    if max_token_length < min_num:
        return min_num

    if max_token_length > max_num:
        return max_num

    power = min_num
    while power < max_token_length:
        power = power * 2
    return power


class ColbertTokenEmbeddings(TokenEmbeddings):
    """
    Colbert embeddings model.

    The embedding runs locally and requires the colbert library to be installed.

    Example:
    Currently the pyarrow module requires a specific version to be installed.

    pip uninstall pyarrow && pip install pyarrow==14.0.0
    pip install colbert-ai==0.2.19
    pip torch

    To take advantage of GPU, please install faiss-gpu
    """

    colbert_config: ColBERTConfig
    checkpoint: Checkpoint
    encoder: CollectionEncoder
    query_tokenizer: QueryTokenizer

    def __init__(
        self,
        checkpoint: str = DEFAULT_COLBERT_MODEL,
        doc_maxlen: int = 220,
        nbits: int = 1,
        kmeans_niters: int = 4,
        nranks: int = -1,
        query_maxlen: int = 32,
        verbose: int = 3,  # 3 is the default on ColBERT checkpoint
        distributed_communication: bool = False,
        **kwargs,
    ):
        self.__cuda = torch.cuda.is_available()
        self.__nranks = reconcile_nranks(nranks)
        total_visible_gpus = torch.cuda.device_count()
        if self.__cuda:
            self.__cuda_device_count = torch.cuda.device_count()
        logging.info(f"run nranks {self.__nranks}")
        if (
            self.__nranks > 1
            and not dist.is_initialized()
            and distributed_communication
        ):
            logging.warn(f"distribution initialization must complete on {nranks} gpus")
            Distributed(self.__nranks)
            logging.info("distribution initialization completed")

        with Run().context(RunConfig(nranks=nranks)):
            if self.__cuda:
                torch.cuda.empty_cache()
            self.colbert_config = ColBERTConfig(
                doc_maxlen=doc_maxlen,
                nbits=nbits,
                kmeans_niters=kmeans_niters,
                nranks=nranks,
                checkpoint=checkpoint,
                query_maxlen=query_maxlen,
                gpus=total_visible_gpus,
            )
        logging.info("creating checkpoint")
        self.checkpoint = Checkpoint(
            self.colbert_config.checkpoint,
            colbert_config=self.colbert_config,
            verbose=verbose,
        )
        self.encoder = CollectionEncoder(
            config=self.colbert_config, checkpoint=self.checkpoint
        )
        self.query_tokenizer = QueryTokenizer(self.colbert_config)
        self.__cuda = torch.cuda.is_available()
        if self.__cuda:
            self.checkpoint = self.checkpoint.cuda()

    def embed_chunks(
        self, texts: List[str], doc_id: Optional[str] = None
    ) -> List[EmbeddedChunk]:
        """Embed search docs."""
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        return self.encode(texts=texts, doc_id=doc_id)

    # this is query embedding without padding
    # it does not reload checkpoint which means faster embedding
    def embed_query(self, query_text: str) -> Tensor:
        chunk_embedding = self.encode(texts=[query_text])[0]
        return chunk_embedding.embeddings

    def encode_queries(
        self,
        query: Union[str, List[str]],
        full_length_search: Optional[bool] = False,
        # query_maxlen is really a fixed length for query token embeddings
        # the length does not grow or shrink despite the number of tokens in the query
        # we continue to use the same term to align with ColBERT documentation/library
        query_maxlen: int = -1,
    ) -> Tensor:
        queries = query if isinstance(query, list) else [query]
        bsize = 128 if len(queries) > 128 else None

        tokens = self.query_tokenizer.tokenize(queries)
        fixed_length = max(query_maxlen, self.colbert_config.query_maxlen)
        if query_maxlen < 0:
            fixed_length = calculate_query_maxlen(
                tokens,
                max(query_maxlen, self.colbert_config.query_maxlen),
                MAX_MODEL_TOKENS,
            )
        # we only send one query at a time therefore tokens[0]
        logging.info(
            f"{len(tokens[0])} tokens in first query with query_maxlen {fixed_length}"
        )

        self.checkpoint.query_tokenizer.query_maxlen = fixed_length

        # All query embeddings in the ColBERT documentation
        # this name, EQ or Q, maps the exact name in most colBERT papers
        queriesQ = self.checkpoint.queryFromText(
            queries, bsize=bsize, to_cpu=True, full_length_search=full_length_search
        )
        return queriesQ

    def encode_query(
        self,
        query: str,
        full_length_search: Optional[bool] = False,
        query_maxlen: int = -1,
    ) -> Tensor:
        queries = self.encode_queries(
            query, full_length_search, query_maxlen=query_maxlen
        )
        return queries[0]

    def encode(
        self, texts: List[str], doc_id: Optional[str] = None
    ) -> List[EmbeddedChunk]:
        runner = Runner(self.__nranks)
        return runner.encode(
            self.colbert_config, texts, doc_id,
        )

