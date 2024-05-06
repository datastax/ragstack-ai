"""
This module integrates the ColBERT model with token embedding functionalities, offering tools for efficiently
encoding queries and text chunks into dense vector representations. It facilitates semantic search and
retrieval by providing optimized methods for embedding generation and manipulation.

The core component, ColbertEmbeddingModel, leverages pre-trained ColBERT models to produce embeddings suitable
for high-relevancy retrieval tasks, with support for both CPU and GPU computing environments.
"""

import logging
import uuid
from typing import Dict, List, Optional

import torch
from torch import Tensor

from colbert.infra import ColBERTConfig, Run, RunConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.modeling.tokenization import QueryTokenizer

from .base_embedding_model import BaseEmbeddingModel
from .constant import DEFAULT_COLBERT_MODEL
from .distributed import ChunkEncoder, Runner
from .objects import Embedding, TextChunk, TextEmbedding


def calculate_query_maxlen(tokens: List[List[str]]) -> int:
    """
    Calculates an appropriate maximum query length for token embeddings, based on the length of the tokenized input.

    Parameters:
        tokens (List[List[str]]): A nested list where each sublist contains tokens from a single query or chunk.

    Returns:
        int: The calculated maximum length for query tokens, adhering to the specified minimum and maximum bounds,
             and adjusted to the nearest power of two.
    """

    max_token_length = max(len(inner_list) for inner_list in tokens)

    # tokens from the query tokenizer does not include the SEP, CLS
    # SEP, CLS, and Q tokens are added to the query
    # although there could be more SEP tokens if there are more than one sentences, we only add one
    return max_token_length + 3


class ColbertEmbeddingModel(BaseEmbeddingModel):
    """
    A class for generating token embeddings using a ColBERT model. This class provides functionalities for
    encoding queries and document chunks into dense vector representations, facilitating semantic search and
    retrieval tasks. It leverages a pre-trained ColBERT model and supports distributed computing environments.

    The class supports both GPU and CPU operations, with GPU usage recommended for performance efficiency.

    Attributes:
        colbert_config (ColBERTConfig): Configuration parameters for the Colbert model.
        checkpoint (Checkpoint): Manages the loading of the model and its parameters.
        query_tokenizer (QueryTokenizer): Tokenizes queries for embedding.
    """

    colbert_config: ColBERTConfig
    checkpoint: Checkpoint
    query_tokenizer: QueryTokenizer

    """
    checkpoint is the where the ColBERT model can be specified or downloaded from huggingface
    colbert_model_url overwrites the checkpoint value if it exists
    doc_maxlen is the number tokens each passage is truncated to
    nbits is the number bits that each dimension encodes to
    kmeans_niters specifies the number of iterations of kmeans clustering
    nrank is the number of processors embeddings can run on
          under the default value of -1, the program runs on all available GPUs under CUDA
    query_maxlen is the fixed length of the tokens for query/recall encoding. Anything less will be padded.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_COLBERT_MODEL,
        doc_maxlen: int = 220,
        nbits: int = 2,
        kmeans_niters: int = 4,
        nranks: int = -1,
        query_maxlen: int = -1,
        verbose: int = 3,  # 3 is the default on ColBERT checkpoint
        multiprocessing_enabled: bool = False,
        **kwargs,
    ):
        """
        Initializes a new instance of the ColbertEmbeddingModel class, setting up the model configuration,
        loading the necessary checkpoints, and preparing the tokenizer and encoder.

        Parameters:
            checkpoint (str): Path or URL to the Colbert model checkpoint. Default is a pre-defined model.
            doc_maxlen (int): Maximum number of tokens for document chunks.
            nbits (int): The number bits that each dimension encodes to.
            kmeans_niters (int): Number of iterations for k-means clustering during quantization.
            nranks (int): Number of ranks (processors) to use for distributed computing; -1 uses all available CPUs/GPUs.
            query_maxlen (int): Maximum length of query tokens for embedding.
            verbose (int): Verbosity level for logging.
            multiprocessing_enabled (bool): Flag to enable distributed computation.
            **kwargs: Additional keyword arguments for future extensions.

        Note:
            This initializer also prepares the system for distributed computation if specified and available.
        """

        self.__cuda = torch.cuda.is_available()
        self.__nranks = nranks
        self.__query_maxlen = query_maxlen

        logging.info(f"Cuda enabled GPU available: {self.__cuda}")
        self.__use_multiprocessing = multiprocessing_enabled and self.__cud

        if self.__use_multiprocessing:
            logging.info("distribution initialization completed")

            total_visible_gpus = torch.cuda.device_count()
            with Run().context(RunConfig(nranks=total_visible_gpus)):
                if self.__cuda:
                    torch.cuda.empty_cache()
                self.colbert_config = ColBERTConfig(
                    doc_maxlen=doc_maxlen,
                    nbits=nbits,
                    kmeans_niters=kmeans_niters,
                    nranks=total_visible_gpus,
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
            self.query_tokenizer = QueryTokenizer(self.colbert_config)
            if self.__cuda:
                self.checkpoint = self.checkpoint.cuda()
        else:
            self.colbert_config = ColBERTConfig(
                doc_maxlen=doc_maxlen,
                nbits=nbits,
                kmeans_niters=kmeans_niters,
                nranks=self.__nranks,
                checkpoint=checkpoint,
                query_maxlen=query_maxlen,
            )
            self.checkpoint = Checkpoint(
                name=self.colbert_config.checkpoint,
                colbert_config=self.colbert_config,
                verbose=verbose,
            )
            self.query_tokenizer = QueryTokenizer(self.colbert_config)

    # implements the Abstract Class Method
    def embed_texts(
        self, texts: List[str]
    ) -> List[Embedding]:
        """
        Embeds a list of texts into their corresponding vector embedding representations.

        Parameters:
            texts (List[str]): A list of string texts.

        Returns:
            List[Embedding]: A list of embeddings, in the order of the input list
        """

        chunks = [TextChunk(index=i, text=t) for i, t in enumerate(texts)]

        if self.__use_multiprocessing:
            timeout = 60 + len(chunks)
            embedded_texts = self._encode_texts_using_multiprocessing(chunks=chunks, timeout=timeout)
        else:
            embedded_texts = self._encode_texts_using_local(chunks=chunks)

        sorted_embedded_texts = sorted(embedded_texts, key=lambda x: x.index)

        return [t.embedding for t in sorted_embedded_texts]

    # implements the Abstract Class Method
    def embed_query(
        self,
        query: str,
        full_length_search: Optional[bool] = False,
        query_maxlen: int = -1,
    ) -> Embedding:
        """
        Embeds a single query text into its vector representation.

        If the query has fewer than query_maxlen tokens it will be padded with BERT special [mast] tokens.

        Parameters:
            query (str): The query string to encode.
            full_length_search (Optional[bool]): Indicates whether to encode the query for a full-length search.
                                                  Defaults to False.
            query_maxlen (int): The fixed length for the query token embedding. If -1, uses a dynamically calculated value.

        Returns:
            Embedding: A vector embedding representation of the query text
        """

        embeddings = self._encode_queries_using_local(
            [query], full_length_search, query_maxlen=query_maxlen
        )
        return embeddings[0]

    def _encode_texts_using_multiprocessing(
        self,
        chunks: List[TextChunk],
        timeout: int = 60,
    ) -> List[TextEmbedding]:
        """
        Encodes a list of texts chunks into embeddings, represented as TextEmbedding objects. This
        method leverages the ColBERT model's encoding capabilities to convert textual content into
        dense vector representations suitable for semantic search and retrieval applications.

        Parameters:
            chunks (List[TextChunk]): The list of text chunks to encode.
            doc_id: An identifier for the document from which the chunks are derived.
            timeout (int): The timeout in seconds for the encoding operation. Defaults to 60 seconds.

        Returns:
            List[TextEmbedding]: A list of TextEmbedding objects containing the embeddings for each text, along
                                  with their associated chunk identifiers.
        """

        runner = Runner(self.__nranks)
        return runner.encode(
            config=self.colbert_config,
            chunks=chunks,
            timeout=timeout,
        )

    def _encode_texts_using_local(
        self,
        chunks: List[TextChunk],
    ) -> List[TextEmbedding]:
        """
        Encodes a list of texts chunks into embeddings, represented as TextEmbedding objects. This
        method leverages the ColBERT model's encoding capabilities to convert textual content into
        dense vector representations suitable for semantic search and retrieval applications.

        Parameters:
            chunks (List[TextChunk]): The list of text chunks to encode.
            doc_id: An identifier for the document from which the chunks are derived.
            timeout (int): The timeout in seconds for the encoding operation. Defaults to 60 seconds.

        Returns:
            List[TextEmbedding]: A list of TextEmbedding objects containing the embeddings for each text, along
                                  with their associated chunk identifiers.
        """
        encoder = ChunkEncoder(config=self.colbert_config)
        return encoder.encode_chunks(chunks=chunks)

    def _encode_queries_using_local(
        self,
        queries: List[str],
        full_length_search: Optional[bool] = False,
        query_maxlen: int = -1,
    ) -> Tensor:
        """
        Encodes one or more texts (queries) into dense vector representations. It supports encoding queries to a fixed
        length, adjusting for the maximum token length or padding as necessary. The method is suitable for both
        single and batch query processing, with optional support for full-length search encoding.

        Parameters:
            queries (List[str]): A single query string or a list of query strings to be encoded.
            full_length_search (Optional[bool]): If True, encodes queries for full-length search. Defaults to False.
            query_maxlen (int): A fixed length for query token embeddings. If -1, uses a dynamically calculated value.

        Returns:
            Tensor: A tensor containing the encoded queries. If multiple queries are provided, the tensor will
                    contain one row per query.
        """

        _query_maxlen = max(query_maxlen, self.colbert_config.query_maxlen)
        if _query_maxlen < 0:
            tokens = self.query_tokenizer.tokenize(queries)
            _query_maxlen = calculate_query_maxlen(tokens)
            logging.debug(f"Calculated dynamic query_maxlen of {_query_maxlen}")

        self.checkpoint.query_tokenizer.query_maxlen = _query_maxlen

        # All query embeddings in the ColBERT documentation
        # this name, EQ or Q, maps the exact name in most colBERT papers
        batch_size = 128 if len(queries) > 128 else None
        to_cpu = not self.__cuda
        queriesQ = self.checkpoint.queryFromText(
            queries, bsize=batch_size, to_cpu=to_cpu, full_length_search=full_length_search
        )
        return queriesQ
