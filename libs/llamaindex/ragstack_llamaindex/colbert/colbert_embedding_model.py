import asyncio

from typing import Any, Coroutine, List, Optional, Tuple

from llama_index.core.async_utils import run_jobs
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.instrumentation.events.base import BaseEvent
from llama_index.core.utils import get_tqdm_iterable

from ragstack_colbert import ColbertEmbeddingModel as RagstackColbertEmbeddingModel
from ragstack_colbert import Embedding, DEFAULT_COLBERT_MODEL

import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)

class ColbertEmbeddingStartEvent(BaseEvent):
    """ColbertEmbeddingStartEvent.

    Args:
        model_dict (dict): Model dictionary containing details about the embedding model.
    """

    model_dict: dict

    @classmethod
    def class_name(cls):
        """Class name."""
        return "ColbertEmbeddingStartEvent"


class ColbertEmbeddingEndEvent(BaseEvent):
    """ColbertEmbeddingEndEvent.

    Args:
        chunks (List[str]): List of chunks.
        embeddings (List[Embedding]): List of embeddings.

    """

    chunks: List[str]
    embeddings: List[Embedding]

    @classmethod
    def class_name(cls):
        """Class name."""
        return "ColbertEmbeddingEndEvent"


class ColbertEmbeddingModel(BaseEmbedding):

    class Config:
        arbitrary_types_allowed = True

    embedding_model: RagstackColbertEmbeddingModel = Field(default=None,
        description="The underlying ragstack_colbert embedding model"
    )

    def __init__(
        self,
        checkpoint: Optional[str] = DEFAULT_COLBERT_MODEL,
        doc_maxlen: Optional[int] = 256,
        nbits: Optional[int] = 2,
        kmeans_niters: Optional[int] = 4,
        nranks: Optional[int] = -1,
        query_maxlen: Optional[int] = None,
        verbose: Optional[int] = 3,  # 3 is the default on ColBERT checkpoint
        chunk_batch_size: Optional[int] = 640,
        **kwargs,
    ):
        """
        Initializes a new instance of the ColbertEmbeddingModel class, setting up the model configuration,
        loading the necessary checkpoints, and preparing the tokenizer and encoder.

        Parameters:
            checkpoint (Optional[str]): Path or URL to the Colbert model checkpoint. Default is a pre-defined model.
            doc_maxlen (Optional[int]): Maximum number of tokens for document chunks. Should equal the chunk_size.
            nbits (Optional[int]): The number bits that each dimension encodes to.
            kmeans_niters (Optional[int]): Number of iterations for k-means clustering during quantization.
            nranks (Optional[int]): Number of ranks (processors) to use for distributed computing; -1 uses all available CPUs/GPUs.
            query_maxlen (Optional[int]): Maximum length of query tokens for embedding. Dynamically calculated if None.
            verbose (Optional[int]): Verbosity level for logging.
            chunk_batch_size (Optional[int]): The number of chunks to batch during embedding. Defaults to 640.
            **kwargs: Additional keyword arguments for future extensions.
        """

        super().__init__(**kwargs)

        self.model_name = checkpoint
        self.embed_batch_size = chunk_batch_size
        self.num_workers = 1
        self.embedding_model = RagstackColbertEmbeddingModel(
            checkpoint=checkpoint,
            doc_maxlen=doc_maxlen,
            nbits=nbits,
            kmeans_niters=kmeans_niters,
            nranks=nranks,
            query_maxlen=query_maxlen,
            verbose=verbose,
            chunk_batch_size=chunk_batch_size,
            **kwargs,
        )



    def _get_query_embedding(
        self,
        query: str,
        full_length_search: Optional[bool] = False,
        query_maxlen: int = None,
    ) -> Embedding:
        """
        Embed the input query synchronously.

        Parameters:
            query (str): The query string to encode.
            full_length_search (Optional[bool]): Indicates whether to encode the query for a full-length search.
                                                  Defaults to False.
            query_maxlen (int): The fixed length for the query token embedding. If None, uses a dynamically calculated value.
        """
        return self.embedding_model.embed_query(query=query, full_length_search=full_length_search, query_maxlen=query_maxlen)

    async def _aget_query_embedding(
        self,
        query: str,
        full_length_search: Optional[bool] = False,
        query_maxlen: int = None,
    ) -> Embedding:
        """
        Embed the input query asynchronously.

         Parameters:
            query (str): The query string to encode.
            full_length_search (Optional[bool]): Indicates whether to encode the query for a full-length search.
                                                  Defaults to False.
            query_maxlen (int): The fixed length for the query token embedding. If None, uses a dynamically calculated value.
        """
        return self.embedding_model.embed_query(query=query, full_length_search=full_length_search, query_maxlen=query_maxlen)

    @dispatcher.span
    def get_query_embedding(self, query: str) -> Embedding:
        """
        Embed the input query.
        """
        dispatch_event = dispatcher.get_dispatch_event()

        model_dict = self.to_dict()
        model_dict.pop("api_key", None)
        dispatch_event(
            ColbertEmbeddingStartEvent(
                model_dict=model_dict,
            )
        )
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            query_embedding = self._get_query_embedding(query)

            event.on_end(
                payload={
                    EventPayload.CHUNKS: [query],
                    EventPayload.EMBEDDINGS: [query_embedding],
                },
            )
        dispatch_event(
            ColbertEmbeddingEndEvent(
                chunks=[query],
                embeddings=[query_embedding],
            )
        )
        return query_embedding

    @dispatcher.span
    async def aget_query_embedding(self, query: str) -> Embedding:
        """Get query embedding."""
        dispatch_event = dispatcher.get_dispatch_event()

        model_dict = self.to_dict()
        model_dict.pop("api_key", None)
        dispatch_event(
            ColbertEmbeddingStartEvent(
                model_dict=model_dict,
            )
        )
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            query_embedding = await self._aget_query_embedding(query)

            event.on_end(
                payload={
                    EventPayload.CHUNKS: [query],
                    EventPayload.EMBEDDINGS: [query_embedding],
                },
            )
        dispatch_event(
            ColbertEmbeddingEndEvent(
                chunks=[query],
                embeddings=[query_embedding],
            )
        )
        return query_embedding

    def _get_text_embedding(self, text: str) -> Embedding:
        """
        Embed the input text synchronously.

        Parameters:
            text (str): A text string to embed.

        Returns:
            Embedding: The embedding vector representation of the input text
        """
        return self.embedding_model.embed_texts(texts=[text])[0]

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """
        Embed the input text asynchronously.

        Parameters:
            text (str): A text string to embed.

        Returns:
            Embedding: The embedding vector representation of the input text
        """
        return self.embedding_model.embed_texts(texts=[text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """
        Embed the input sequence of text synchronously.

        Parameters:
            texts (List[str]): A list of string texts.

        Returns:
            List[Embedding]: A list of embeddings, in the order of the input list
        """
        return self.embedding_model.embed_texts(texts=texts)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """
        Embed the input sequence of text asynchronously.

        Parameters:
            texts (List[str]): A list of string texts.

        Returns:
            List[Embedding]: A list of embeddings, in the order of the input list
        """
        return self.embedding_model.embed_texts(texts=texts)

    @dispatcher.span
    def get_text_embedding(self, text: str) -> Embedding:
        """
        Embed the input text.

        When embedding text, depending on the model, a special instruction
        can be prepended to the raw text string. For example, "Represent the
        document for retrieval: ". If you're curious, other examples of
        predefined instructions can be found in embeddings/huggingface_utils.py.
        """
        dispatch_event = dispatcher.get_dispatch_event()

        model_dict = self.to_dict()
        model_dict.pop("api_key", None)
        dispatch_event(
            ColbertEmbeddingStartEvent(
                model_dict=model_dict,
            )
        )
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            text_embedding = self._get_text_embedding(text)

            event.on_end(
                payload={
                    EventPayload.CHUNKS: [text],
                    EventPayload.EMBEDDINGS: [text_embedding],
                }
            )
        dispatch_event(
            ColbertEmbeddingEndEvent(
                chunks=[text],
                embeddings=[text_embedding],
            )
        )
        return text_embedding

    @dispatcher.span
    async def aget_text_embedding(self, text: str) -> Embedding:
        """Async get text embedding."""
        dispatch_event = dispatcher.get_dispatch_event()

        model_dict = self.to_dict()
        model_dict.pop("api_key", None)
        dispatch_event(
            ColbertEmbeddingStartEvent(
                model_dict=model_dict,
            )
        )
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            text_embedding = await self._aget_text_embedding(text)

            event.on_end(
                payload={
                    EventPayload.CHUNKS: [text],
                    EventPayload.EMBEDDINGS: [text_embedding],
                }
            )
        dispatch_event(
            ColbertEmbeddingEndEvent(
                chunks=[text],
                embeddings=[text_embedding],
            )
        )
        return text_embedding

    @dispatcher.span
    def get_text_embedding_batch(
        self,
        texts: List[str],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[Embedding]:
        """Get a list of text embeddings, with batching."""
        dispatch_event = dispatcher.get_dispatch_event()

        cur_batch: List[str] = []
        result_embeddings: List[Embedding] = []

        queue_with_progress = enumerate(
            get_tqdm_iterable(texts, show_progress, "Generating embeddings")
        )

        model_dict = self.to_dict()
        model_dict.pop("api_key", None)
        for idx, text in queue_with_progress:
            cur_batch.append(text)
            if idx == len(texts) - 1 or len(cur_batch) == self.embed_batch_size:
                # flush
                dispatch_event(
                    ColbertEmbeddingStartEvent(
                        model_dict=model_dict,
                    )
                )
                with self.callback_manager.event(
                    CBEventType.EMBEDDING,
                    payload={EventPayload.SERIALIZED: self.to_dict()},
                ) as event:
                    embeddings = self._get_text_embeddings(cur_batch)
                    result_embeddings.extend(embeddings)
                    event.on_end(
                        payload={
                            EventPayload.CHUNKS: cur_batch,
                            EventPayload.EMBEDDINGS: embeddings,
                        },
                    )
                dispatch_event(
                    ColbertEmbeddingEndEvent(
                        chunks=cur_batch,
                        embeddings=embeddings,
                    )
                )
                cur_batch = []

        return result_embeddings

    @dispatcher.span
    async def aget_text_embedding_batch(
        self, texts: List[str], show_progress: bool = False
    ) -> List[Embedding]:
        """Asynchronously get a list of text embeddings, with batching."""
        dispatch_event = dispatcher.get_dispatch_event()
        num_workers = self.num_workers

        model_dict = self.to_dict()
        model_dict.pop("api_key", None)

        cur_batch: List[str] = []
        callback_payloads: List[Tuple[str, List[str]]] = []
        result_embeddings: List[Embedding] = []
        embeddings_coroutines: List[Coroutine] = []
        for idx, text in enumerate(texts):
            cur_batch.append(text)
            if idx == len(texts) - 1 or len(cur_batch) == self.embed_batch_size:
                # flush
                dispatch_event(
                    ColbertEmbeddingStartEvent(
                        model_dict=model_dict,
                    )
                )
                event_id = self.callback_manager.on_event_start(
                    CBEventType.EMBEDDING,
                    payload={EventPayload.SERIALIZED: self.to_dict()},
                )
                callback_payloads.append((event_id, cur_batch))
                embeddings_coroutines.append(self._aget_text_embeddings(cur_batch))
                cur_batch = []

        # flatten the results of asyncio.gather, which is a list of embeddings lists
        nested_embeddings = []

        if num_workers and num_workers > 1:
            nested_embeddings = await run_jobs(
                embeddings_coroutines,
                show_progress=show_progress,
                workers=self.num_workers,
                desc="Generating embeddings",
            )
        else:
            if show_progress:
                try:
                    from tqdm.asyncio import tqdm_asyncio

                    nested_embeddings = await tqdm_asyncio.gather(
                        *embeddings_coroutines,
                        total=len(embeddings_coroutines),
                        desc="Generating embeddings",
                    )
                except ImportError:
                    nested_embeddings = await asyncio.gather(*embeddings_coroutines)
            else:
                nested_embeddings = await asyncio.gather(*embeddings_coroutines)

        result_embeddings = [
            embedding for embeddings in nested_embeddings for embedding in embeddings
        ]

        for (event_id, text_batch), embeddings in zip(
            callback_payloads, nested_embeddings
        ):
            dispatch_event(
                ColbertEmbeddingEndEvent(
                    chunks=text_batch,
                    embeddings=embeddings,
                )
            )
            self.callback_manager.on_event_end(
                CBEventType.EMBEDDING,
                payload={
                    EventPayload.CHUNKS: text_batch,
                    EventPayload.EMBEDDINGS: embeddings,
                },
                event_id=event_id,
            )

        return result_embeddings