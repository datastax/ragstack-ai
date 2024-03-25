from typing import List, Union
import logging
import itertools
import torch
from torch import Tensor
import uuid
from .token_embedding import TokenEmbeddings, PerTokenEmbeddings, PassageEmbeddings
from .constant import MAX_MODEL_TOKENS, DEFAULT_COLBERT_MODEL

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.modeling.checkpoint import Checkpoint
from colbert.modeling.tokenization import QueryTokenizer


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
    ):
        self.__cuda = torch.cuda.is_available()
        total_visible_gpus = 0
        if self.__cuda:
            self.__cuda_device_count = torch.cuda.device_count()
            self.__cuda_device_name = torch.cuda.get_device_name()
            logging.info(f"nrank {nranks}")
            if nranks < 1:
                nranks = self.__cuda_device_count
            if nranks > 1:
                total_visible_gpus = self.__cuda_device_count
            logging.info(
                f"run on {self.__cuda_device_count} gpus and visible {total_visible_gpus} gpus embeddings on {nranks} gpus"
            )
        else:
            if nranks < 1:
                nranks = 1

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
        self.__doc_maxlen = doc_maxlen
        self.__nbits = nbits
        self.__kmeans_niters = kmeans_niters
        self.__nranks = nranks
        logging.info("creating checkpoint")
        self.checkpoint = Checkpoint(
            self.colbert_config.checkpoint, colbert_config=self.colbert_config
        )
        self.encoder = CollectionEncoder(
            config=self.colbert_config, checkpoint=self.checkpoint
        )
        self.query_tokenizer = QueryTokenizer(self.colbert_config)
        self.__cuda = torch.cuda.is_available()
        if self.__cuda:
            self.checkpoint = self.checkpoint.cuda()

    def embed_documents(
        self, texts: List[str], doc_id: str = ""
    ) -> List[PassageEmbeddings]:
        """Embed search docs."""
        return self.encode(texts, doc_id)

    # this is query embedding without padding
    # it does not reload checkpoint which means faster embedding
    def embed_query(self, text: str) -> Tensor:
        passage_embeddings = self.encode(texts=[text], doc_id="no-op")[0]
        embeddings = []
        for token in passage_embeddings.get_all_token_embeddings():
            embeddings.append(token.get_embeddings())

        return torch.tensor(embeddings)

    def encode_queries(
        self,
        query: Union[str, List[str]],
        full_length_search: bool = False,
        # query_maxlen is really a fixed length for query token embeddings
        # the length does not grow or shrink despite the number of tokens in the query
        # we continue to use the same term to align with ColBERT documentation/library
        query_maxlen: int = -1,
    ):
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
        full_length_search: bool = False,
        query_maxlen: int = -1,
    ):
        queries = self.encode_queries(
            query, full_length_search, query_maxlen=query_maxlen
        )
        return queries[0]

    def encode(self, texts: List[str], doc_id: str = "") -> List[PassageEmbeddings]:
        embeddings, count = self.encoder.encode_passages(texts)

        if doc_id == "":
            doc_id = str(uuid.uuid4())

        collection_embeds = []
        # split up embeddings by counts, a list of the number of tokens in each passage
        start_indices = [0] + list(itertools.accumulate(count[:-1]))
        embeddings_by_part = [
            embeddings[start : start + count]
            for start, count in zip(start_indices, count)
        ]
        for part_id, embedding in enumerate(embeddings_by_part):
            passage_embeddings = PassageEmbeddings(
                text=texts[part_id], doc_id=doc_id, part_id=part_id
            )

            for embedding_id, perTokenEmbedding in enumerate(embedding):
                per_token = PerTokenEmbeddings(
                    embedding_id=embedding_id, part_id=part_id
                )
                per_token.add_embeddings(perTokenEmbedding.tolist())
                passage_embeddings.add_token_embeddings(per_token)
            collection_embeds.append(passage_embeddings)

        return collection_embeds
