from __future__ import annotations

try:
    # Try importing the function from itertools (Python 3.12+)
    from itertools import batched  # type: ignore[attr-defined]
except ImportError:
    from itertools import islice
    from typing import TYPE_CHECKING, TypeVar

    if TYPE_CHECKING:
        from collections.abc import Iterable, Iterator

    # Fallback implementation for older Python versions

    T = TypeVar("T")

    # This is equivalent to `itertools.batched`, but that is only available in 3.12
    def batched(iterable: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
        """Emulate itertools.batched."""
        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch


__all__ = ["batched"]
