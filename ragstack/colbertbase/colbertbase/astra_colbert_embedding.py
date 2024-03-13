from typing import Any, Dict, List, Union
import datetime
import itertools
import torch  # it should part of colbert dependencies
import uuid
import os
from .token_embedding import TokenEmbeddings, PerTokenEmbeddings, PassageEmbeddings


from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.modeling.checkpoint import Checkpoint


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

    # these are default values aligned with the colbert library
    __doc_maxlen: int = 220,
    __nbits: int = 1,
    __kmeans_niters: int = 4,
    __nranks: int = 1,
    __index_bsize: int = 64,

    # TODO: expose these values
    # these are default values aligned with the colbert library
    __resume: bool = False,
    __similarity: str = 'cosine',
    __bsize: int = 32,
    __accumsteps: int = 1,
    __lr: float = 0.000003,
    __maxsteps: int = 500000,
    __nway: int = 2,
    __use_ib_negatives: bool = False,
    __reranker: bool = False,
    __is_cuda: bool = False

    @classmethod
    def validate_environment(self, values: Dict) -> Dict:
        """Validate colbert and its dependency is installed."""
        try:
            import torch
            if torch.cuda.is_available():
                self.__is_cuda = True

        except ImportError as exc:
            raise ImportError(
                "Could not import torch library. "
                "Please install it with `pip install torch`"
            ) from exc

        return values

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
            print(f"nrank {nranks}")
            if nranks < 1:
                nranks = self.__cuda_device_count
            if nranks > 1:
                total_visible_gpus = self.__cuda_device_count
            print(f"run on {self.__cuda_device_count} gpus and visible {total_visible_gpus} gpus embeddings on {nranks} gpus")
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
        print("creating checkpoint")
        self.checkpoint = Checkpoint(self.colbert_config.checkpoint, colbert_config=self.colbert_config)
        self.encoder = CollectionEncoder(config=self.colbert_config, checkpoint=self.checkpoint)
        self.__cuda = torch.cuda.is_available()
        if self.__cuda:
            self.checkpoint = self.checkpoint.cuda()

        self.print_memory_stats("ColbertTokenEmbeddings")

    def embed_documents(self, texts: List[str], title: str="") -> List[PassageEmbeddings]:
        if title == "":
            title = str(uuid.uuid4())
        """Embed search docs."""
        return self.encode(texts, title)

    def embed_query(self, text: str, title: str) -> PassageEmbeddings:
        return self.embed_documents([text], title)[0]

    def encode_queries(
            self,
            query: Union[str, List[str]],
            full_length_search: bool = False,
            query_maxlen: int = 32,
        ):
        queries = query if type(query) is list else [query]
        bsize = 128 if len(queries) > 128 else None

        self.checkpoint.query_tokenizer.query_maxlen = max(query_maxlen, self.colbert_config.query_maxlen)
        Q = self.checkpoint.queryFromText(queries, bsize=bsize, to_cpu=True, full_length_search=full_length_search)

        return Q

    def encode_query(
            self,
            query: str,
            full_length_search: bool = False,
            query_maxlen: int = 32,
        ):
        Q = self.encode_queries(query, full_length_search, query_maxlen=query_maxlen)
        return Q[0]

    def encode(self, texts: List[str], title: str="") -> List[PassageEmbeddings]:
        # collection = Collection(texts)
        # batches = collection.enumerate_batches(rank=Run().rank)
        '''
        config = ColBERTConfig(
            doc_maxlen=self.__doc_maxlen,
            nbits=self.__nbits,
            kmeans_niters=self.__kmeans_niters,
            checkpoint=self.checkpoint,
            index_bsize=1)
        ckp = Checkpoint(config.checkpoint, colbert_config=config)
        encoder = CollectionEncoder(config=self.config, checkpoint=self.checkpoint)
        '''
        embeddings, count = self.encoder.encode_passages(texts)

        collectionEmbds = []
        # split up embeddings by counts, a list of the number of tokens in each passage
        start_indices = [0] + list(itertools.accumulate(count[:-1]))
        embeddings_by_part = [embeddings[start:start+count] for start, count in zip(start_indices, count)]
        for part, embedding in enumerate(embeddings_by_part):
            collectionEmbd = PassageEmbeddings(text=texts[part], title=title, part=part)
            pid = collectionEmbd.id()
            for __part_i, perTokenEmbedding in enumerate(embedding):
                perToken = PerTokenEmbeddings(parent_id=pid, id=__part_i, title=title, part=part)
                perToken.add_embeddings(perTokenEmbedding.tolist())
                # print(f"    token embedding part {part} id {__part_i} parent id {pid}")
                collectionEmbd.add_token_embeddings(perToken)
            collectionEmbds.append(collectionEmbd)
            # print(f"embedding part {part} collection id {pid}, collection size {len(collectionEmbd.get_all_token_embeddings())}")

        return collectionEmbds

    def print_message(self, *s, condition=True, pad=False):
        s = ' '.join([str(x) for x in s])
        msg = "[{}] {}".format(datetime.datetime.now().strftime("%b %d, %H:%M:%S"), s)

        if condition:
            msg = msg if not pad else f'\n{msg}\n'
            print(msg, flush=True)

        return msg

    def print_memory_stats(self, message=''):
        try:
            import psutil  # Remove before releases? Or at least make optional with try/except.
        except ImportError:
            self.print_message("psutil not installed. Memory stats not available.")
            return

        global_info = psutil.virtual_memory()
        total, available, used, free = global_info.total, global_info.available, global_info.used, global_info.free

        info = psutil.Process().memory_info()
        rss, vms, shared = info.rss, info.vms, info.shared
        uss = psutil.Process().memory_full_info().uss

        gib = 1024 ** 3

        summary = f"""
        "[PID: {os.getpid()}]
        [{message}]
        Available: {available / gib:,.1f} / {total / gib:,.1f}
        Free: {free / gib:,.1f} / {total / gib:,.1f}
        Usage: {used / gib:,.1f} / {total / gib:,.1f}

        RSS: {rss  / gib:,.1f}
        VMS: {vms  / gib:,.1f}
        USS: {uss  / gib:,.1f}
        SHARED: {shared  / gib:,.1f}
        """.strip().replace('\n', '\t')

        self.print_message(summary, pad=True)
