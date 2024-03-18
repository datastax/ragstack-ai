from typing import Any, List, Union
import logging
import itertools
import torch  # it should part of colbert dependencies
from torch import Tensor
import uuid
from .token_embedding import TokenEmbeddings, PerTokenEmbeddings, PassageEmbeddings
from .constant import MAX_MODEL_TOKENS


from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.modeling.checkpoint import Checkpoint
from colbert.modeling.tokenization import QueryTokenizer

def calculate_query_maxlen(
        tokens: List[List[str]], min_num: int, max_num: int
    ) -> int:
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

    # these are default values aligned with the colbert library
    __doc_maxlen: int = (220,)
    __nbits: int = (1,)
    __kmeans_niters: int = (4,)
    __nranks: int = (1,)
    __index_bsize: int = (64,)

    # TODO: expose these values
    # these are default values aligned with the colbert library
    __resume: bool = (False,)
    __similarity: str = ("cosine",)
    __bsize: int = (32,)
    __accumsteps: int = (1,)
    __lr: float = (0.000003,)
    __maxsteps: int = (500000,)
    __nway: int = (2,)
    __use_ib_negatives: bool = (False,)
    __reranker: bool = (False,)
    __is_cuda: bool = False

    def __init__(
        self,
        checkpoint: str = "colbert-ir/colbertv2.0",
        doc_maxlen: int = 220,
        nbits: int = 1,
        kmeans_niters: int = 4,
        nranks: int = -1,
        query_maxlen: int = 32,
        **data: Any,
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
        self, texts: List[str], title: str = ""
    ) -> List[PassageEmbeddings]:
        if title == "":
            title = str(uuid.uuid4())
        """Embed search docs."""
        return self.encode(texts, title)

    # this is query embedding without padding
    # it does not reload checkpoint which means faster embedding
    def embed_query(self, text: str) -> Tensor:
        passageEmbeddings = self.embed_documents(texts=[text], title="no-op")[0]
        embeddings = []
        for token in passageEmbeddings.get_all_token_embeddings():
            embeddings.append(token.get_embeddings())

        return torch.tensor(embeddings)

    def encode_queries(
        self,
        query: Union[str, List[str]],
        full_length_search: bool = False,
        query_maxlen: int = 32,
        enable_dynamic_query_maxlen: bool = True,
    ):
        queries = query if isinstance(query, list) else [query]
        bsize = 128 if len(queries) > 128 else None

        tokens = self.query_tokenizer.tokenize(queries)
        logging.info(f"query token size {len(tokens)}, tokens {tokens}")
        fixed_length = query_maxlen
        if enable_dynamic_query_maxlen:
            fixed_length = calculate_query_maxlen(
                tokens,
                max(query_maxlen, self.colbert_config.query_maxlen),
                MAX_MODEL_TOKENS)
        logging.info(f"query token size {fixed_length}")

        self.checkpoint.query_tokenizer.query_maxlen = fixed_length

        # All query embeddings in the ColBERT documentation
        # this name, EQ or Q, maps the exact name in most colBERT papers
        Q = self.checkpoint.queryFromText(
            queries, bsize=bsize, to_cpu=True, full_length_search=full_length_search
        )

        return Q

    def encode_query(
        self,
        query: str,
        full_length_search: bool = False,
        query_maxlen: int = 32,
    ):
        Q = self.encode_queries(query, full_length_search, query_maxlen=query_maxlen)
        return Q[0]

    def encode(self, texts: List[str], title: str = "") -> List[PassageEmbeddings]:
        embeddings, count = self.encoder.encode_passages(texts)

        collectionEmbds = []
        # split up embeddings by counts, a list of the number of tokens in each passage
        start_indices = [0] + list(itertools.accumulate(count[:-1]))
        embeddings_by_part = [
            embeddings[start : start + count]
            for start, count in zip(start_indices, count)
        ]
        for part, embedding in enumerate(embeddings_by_part):
            collectionEmbd = PassageEmbeddings(text=texts[part], title=title, part=part)
            pid = collectionEmbd.id()
            for __part_i, perTokenEmbedding in enumerate(embedding):
                perToken = PerTokenEmbeddings(
                    parent_id=pid, id=__part_i, title=title, part=part
                )
                perToken.add_embeddings(perTokenEmbedding.tolist())
                # logging.info(f"    token embedding part {part} id {__part_i} parent id {pid}")
                collectionEmbd.add_token_embeddings(perToken)
            collectionEmbds.append(collectionEmbd)
            # logging.info(f"embedding part {part} collection id {pid}, collection size {len(collectionEmbd.get_all_token_embeddings())}")

        return collectionEmbds
