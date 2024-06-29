import dataclasses
from typing import Dict, Iterable, List, Optional

import numpy as np

from ragstack_knowledge_store.math import cosine_similarity


def _emb_to_ndarray(embedding: List[float]) -> np.ndarray:
    embedding = np.array(embedding, dtype=np.float32)
    if embedding.ndim == 1:
        embedding = np.expand_dims(embedding, axis=0)
    return embedding


NEG_INF = float("-inf")


@dataclasses.dataclass
class _Candidate:
    id: str
    weighted_similarity: float
    weighted_redundancy: float
    score: float = dataclasses.field(init=False)

    def __post_init__(self):
        self.score = self.weighted_similarity - self.weighted_redundancy

    def update_redundancy(self, new_weighted_redundancy: float):
        if new_weighted_redundancy > self.weighted_redundancy:
            self.weighted_redundancy = new_weighted_redundancy
            self.score = self.weighted_similarity - self.weighted_redundancy


class MmrHelper:
    dimensions: int
    """Dimensions of the embedding."""

    query_embedding: np.ndarray
    """Embedding of the query as a (1,dim) ndarray."""

    lambda_mult: float
    """Number between 0 and 1.

    Determines the degree of diversity among the results with 0 corresponding to
    maximum diversity and 1 to minimum diversity."""

    lambda_mult_complement: float
    """1 - lambda_mult."""

    score_threshold: float
    """Only documents with a score greater than or equal to this will be chosen."""

    selected_ids: List[str]
    """List of selected IDs (in selection order)."""
    selected_embeddings: np.ndarray
    """(N, dim) ndarray with a row for each selected node."""

    candidate_id_to_index: Dict[str, int]
    """Dictionary of candidate IDs to indices in candidates and candidate_embeddings."""
    candidates: List[_Candidate]
    """List containing information about candidates.

    Same order as rows in `candidate_embeddings`.
    """
    candidate_embeddings: np.ndarray
    """(N, dim) ndarray with a row for each candidate."""

    best_score: float
    best_id: Optional[str]

    def __init__(
        self,
        k: int,
        query_embedding: List[float],
        lambda_mult: float = 0.5,
        score_threshold: float = NEG_INF,
    ) -> None:
        """Create a helper for executing an MMR traversal query.

        Args:
            query_embedding: The embedding of the query to use for scoring.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding to maximum
                diversity and 1 to minimum diversity. Defaults to 0.5.
            score_threshold: Only documents with a score greater than or equal
                this threshold will be chosen. Defaults to -infinity.
        """
        self.query_embedding = _emb_to_ndarray(query_embedding)
        self.dimensions = self.query_embedding.shape[1]

        self.lambda_mult = lambda_mult
        self.lambda_mult_complement = 1 - lambda_mult
        self.score_threshold = score_threshold

        self.selected_ids = []

        # List of selected embeddings (in selection order).
        self.selected_embeddings = np.ndarray((k, self.dimensions), dtype=np.float32)

        self.candidate_id_to_index = {}

        # List of the candidates.
        self.candidates = []
        # ND array of the candidate embeddings.
        self.candidate_embeddings = np.ndarray((0, self.dimensions), dtype=np.float32)

        self.best_score = NEG_INF
        self.best_id = None

    def candidate_ids(self) -> Iterable[str]:
        return self.candidate_id_to_index.keys()

    def _already_selected_embeddings(self) -> np.ndarray:
        """Return the selected embeddings sliced to the already assigned values."""
        selected = len(self.selected_ids)
        return np.vsplit(self.selected_embeddings, [selected])[0]

    def _pop_candidate(self, id: str) -> np.ndarray:
        """Pop the candidate with the given ID.

        Returns:
            The embedding of the candidate.
        """
        # Get the embedding for the id.
        index = self.candidate_id_to_index.pop(id)
        assert self.candidates[index].id == id
        embedding = self.candidate_embeddings[index].copy()

        # Swap that index with the last index in the candidates and
        # candidate_embeddings.
        last_index = self.candidate_embeddings.shape[0] - 1

        if index == last_index:
            # Already the last item. We don't need to swap.
            self.candidates.pop()
        else:
            self.candidate_embeddings[index] = self.candidate_embeddings[last_index]

            old_last = self.candidates.pop()
            self.candidates[index] = old_last
            self.candidate_id_to_index[old_last.id] = index

        self.candidate_embeddings = np.vsplit(self.candidate_embeddings, [last_index])[
            0
        ]

        return embedding

    def pop_best(self) -> Optional[str]:
        """Select and pop the best item being considered.

        Updates the consideration set based on it.

        Returns:
            A tuple containing the ID of the best item.
        """
        if self.best_id is None or self.best_score < self.score_threshold:
            return None

        # Get the selection and remove from candidates.
        selected_id = self.best_id
        selected_embedding = self._pop_candidate(selected_id)

        # Add the ID and embedding to the selected information.
        selection_index = len(self.selected_ids)
        self.selected_ids.append(selected_id)
        self.selected_embeddings[selection_index] = selected_embedding

        # Reset the best score / best ID.
        self.best_score = NEG_INF
        self.best_id = None

        # Update the candidates redundancy, tracking the best node.
        if self.candidate_embeddings.shape[0] > 0:
            similarity = cosine_similarity(
                self.candidate_embeddings, np.expand_dims(selected_embedding, axis=0)
            )
            for index, candidate in enumerate(self.candidates):
                candidate.update_redundancy(similarity[index][0])
                if candidate.score > self.best_score:
                    self.best_score = candidate.score
                    self.best_id = candidate.id

        return selected_id

    def add_candidates(self, candidates: Dict[str, List[float]]):
        """Add candidates to the consideration set."""

        # Determine the keys to actually include.
        # These are the candidates that aren't already selected
        # or under consideration.
        include_ids = set(candidates.keys())
        include_ids.difference_update(self.selected_ids)
        include_ids.difference_update(self.candidate_id_to_index.keys())
        include_ids = list(include_ids)

        # Now, build up a matrix of the remaining candidate embeddings.
        # And add them to the
        new_embeddings = np.ndarray((len(include_ids), self.dimensions))
        offset = self.candidate_embeddings.shape[0]
        for index, id in enumerate(include_ids):
            if id in include_ids:
                self.candidate_id_to_index[id] = offset + index
                embedding = candidates[id]
                new_embeddings[index] = embedding

        # Compute the similarity to the query.
        similarity = cosine_similarity(new_embeddings, self.query_embedding)

        # Compute the distance metrics of all of pairs in the selected set with
        # the new candidates.
        redundancy = cosine_similarity(
            new_embeddings, self._already_selected_embeddings()
        )
        for index, id in enumerate(include_ids):
            max_redundancy = 0.0
            if redundancy.shape[0] > 0:
                max_redundancy = redundancy[index].max()
            candidate = _Candidate(
                id=id,
                weighted_similarity=self.lambda_mult * similarity[index][0],
                weighted_redundancy=self.lambda_mult_complement * max_redundancy,
            )
            self.candidates.append(candidate)

            if candidate.score >= self.best_score:
                self.best_score = candidate.score
                self.best_id = candidate.id

        # Add the new embeddings to the candidate set.
        self.candidate_embeddings = np.vstack(
            (self.candidate_embeddings, new_embeddings)
        )
