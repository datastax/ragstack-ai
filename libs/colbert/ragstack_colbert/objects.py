"""
This module defines a set of data classes for handling chunks of text in various stages of
processing within the ColBERT retrieval system.
"""

from dataclasses import dataclass, field
from numbers import Number
from typing import Any, Dict, List

from torch import Tensor


@dataclass(frozen=True)
class BaseChunk:
    """
    The base class a chunk of text from a document

    Attributes:
        doc_id (str): The document id from which this chunk originates.
        chunk_id (int): The id of this chunk.
    """

    doc_id: str
    chunk_id: int


@dataclass(frozen=True)
class ChunkData():
    """
    Represents a chunk of text from a document including any associated metadata

    Attributes:
        text (str): The text content of this chunk.
        metadata (dict): The metadata of this chunk.
    """

    text: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class EmbeddedChunk(BaseChunk):
    """
    Extends BaseChunk with the ColBERT embedding for the chunk's text.

    Attributes:
        data (ChunkData): Contains the chunk text and metadata
        embeddings (Tensor): A tensor representing the embeddings of the chunk's text. The dimensions
                              are 'the count of tokens in the chunk' by 'the Colbert embedding size
                              per chunk (default 128)'

    Inherits from:
        BaseChunk: Inherits all attributes and methods from the BaseChunk class.
    """

    data: ChunkData
    embeddings: Tensor


    def __len__(self):
        """
        Returns the length of the embeddings tensor, representing the number of dimensions
        in the embedded space.

        Returns:
            int: The number of dimensions in the embeddings tensor.
        """
        return len(self.embeddings)


@dataclass(frozen=True)
class RetrievedChunk(BaseChunk):
    """
    Represents a chunk of text that has been retrieved, including ranking and scoring information.

    Attributes:
        data (ChunkData): Contains the chunk text and metadata
        rank (int): The rank of this chunk in the context of ColBERT retrieval, with a lower number
                    indicating a higher relevance or quality.
        score (Number): The score assigned to this chunk by the ColBERT retrieval system, indicating
                        its relevancy. Higher scores are better.

    Inherits from:
        BaseChunk: Inherits all attributes and methods from the BaseChunk class.
    """

    data: ChunkData
    rank: int
    score: Number

@dataclass(frozen=True)
class BaseText:
    original_index: int
    text: str

@dataclass(frozen=True)
class EmbeddedText(BaseText):
    chunk_id: int
    embeddings: List[Tensor]
