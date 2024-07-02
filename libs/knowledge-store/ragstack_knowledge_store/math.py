"""Copied from langchain_community.utils.math
See https://github.com/langchain-ai/langchain/blob/langchain-community%3D%3D0.0.38/libs/community/langchain_community/utils/math.py
"""

import logging
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)

Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]


def cosine_similarity(x: Matrix, y: Matrix) -> np.ndarray:
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

        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        z = 1 - np.array(simd.cdist(x, y, metric="cosine"))
        return z
    except ImportError:
        logger.debug(
            "Unable to import simsimd, defaulting to NumPy implementation. If you want "
            "to use simsimd please install with `pip install simsimd`."
        )
        x_norm = np.linalg.norm(x, axis=1)
        y_norm = np.linalg.norm(y, axis=1)
        # Ignore divide by zero errors run time warnings as those are handled below.
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(x, y.T) / np.outer(x_norm, y_norm)
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity
