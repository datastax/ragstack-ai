try:
    # Try importing the function from itertools (Python 3.12+)
    from itertools import batched
except ImportError:
    from itertools import islice
    from typing import Iterable, Iterator, TypeVar

    # Fallback implementation for older Python versions

    T = TypeVar("T")

    # This is equivalent to `itertools.batched`, but that is only available in 3.12
    def batched(iterable: Iterable[T], n: int) -> Iterator[Iterator[T]]:
        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch
