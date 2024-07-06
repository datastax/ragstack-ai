"""Copied from langchain_community.utils.math.

See https://github.com/langchain-ai/langchain/blob/langchain-community%3D%3D0.0.38/libs/community/langchain_community/utils/math.py .
"""  # noqa: E501

import logging
from typing import List, Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

Matrix = Union[List[List[float]], List[NDArray[np.float32]], NDArray[np.float32]]


def cosine_similarity(x: Matrix, y: Matrix) -> NDArray[np.float32]:
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(x) == 0 or len(y) == 0:
        return np.array([])

    x = np.array(x)
    y = np.array(y)
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {x.shape} "
            f"and Y has shape {y.shape}."
        )
    try:
        import simsimd as simd
    except ImportError:
        logger.debug(
            "Unable to import simsimd, defaulting to NumPy implementation. If you want "
            "to use simsimd please install with `pip install simsimd`."
        )
        x_norm = np.linalg.norm(x, axis=1)
        y_norm = np.linalg.norm(y, axis=1)
        # Ignore divide by zero errors run time warnings as those are handled below.
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity: NDArray[np.float32] = np.dot(x, y.T) / np.outer(x_norm, y_norm)
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity
    else:
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return 1.0 - np.array(simd.cdist(x, y, metric="cosine"))
