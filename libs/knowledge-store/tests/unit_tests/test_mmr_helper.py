import math
from typing import List

from ragstack_knowledge_store._mmr_helper import MmrHelper


def test_mmr_helper_add_candidate():
    helper = MmrHelper(5, [0.0, 1.0])
    helper.add_candidates(
        {
            "a": [0.0, 1.0],
            "b": [1.0, 0.0],
        }
    )

    assert helper.best_id == "a"


def test_mmr_helper_pop_best():
    helper = MmrHelper(5, [0.0, 1.0])
    helper.add_candidates(
        {
            "a": [0.0, 1.0],
            "b": [1.0, 0.0],
        }
    )
    assert helper.pop_best() == "a"
    assert helper.pop_best() == "b"
    assert helper.pop_best() is None


def angular_embedding(angle: float) -> List[float]:
    return [math.cos(angle * math.pi), math.sin(angle * math.pi)]


def test_mmr_helper_added_documetns():
    """Test end to end construction and MMR search.
    The embedding function used here ensures `texts` become
    the following vectors on a circle (numbered v0 through v3):

           ______ v2
          /      \
         /        |  v1
    v3  |     .    | query
         |        /  v0
          |______/                 (N.B. very crude drawing)


    With fetch_k==2 and k==2, when query is at 0.0, (1, ),
    one expects that v2 and v0 are returned (in some order)
    because v1 is "too close" to v0 (and v0 is closer than v1)).

    Both v2 and v3 are discovered after v0.
    """
    helper = MmrHelper(5, angular_embedding(0.0))

    # Fetching the 2 nearest neighbors to 0.0
    helper.add_candidates(
        {
            "v0": angular_embedding(-0.124),
            "v1": angular_embedding(+0.127),
        }
    )
    assert helper.pop_best() == "v0"

    # After v0 is seletected, new nodes are discovered.
    # v2 is closer than v3. v1 is "too similar" to "v0" so it's not included.
    helper.add_candidates(
        {
            "v2": angular_embedding(+0.25),
            "v3": angular_embedding(+1.0),
        }
    )
    assert helper.pop_best() == "v2"
