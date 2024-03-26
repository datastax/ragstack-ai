from dataclasses import dataclass
from numbers import Number

from torch import Tensor


@dataclass(frozen=True)
class BaseChunk:
    doc_id: str
    chunk_id: int
    text: str


@dataclass(frozen=True)
class EmbeddedChunk(BaseChunk):
    vectors: Tensor

    def __len__(self):
        return len(self.vectors)


@dataclass(frozen=True)
class RetrievedChunk(BaseChunk):
    rank: int
    score: Number
