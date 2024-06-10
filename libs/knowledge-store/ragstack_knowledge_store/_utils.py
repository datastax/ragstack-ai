import sys

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

# TODO: Remove the "polyfill" when we required python is >= 3.10.

if sys.version_info >= (3, 10):

    def strict_zip(*iterables):
        return zip(*iterables, strict=True)
else:

    def strict_zip(*iterables):
        # Custom implementation for Python versions older than 3.10
        if not iterables:
            return

        iterators = tuple(iter(iterable) for iterable in iterables)
        try:
            while True:
                items = []
                for iterator in iterators:
                    items.append(next(iterator))
                yield tuple(items)
        except StopIteration:
            pass

        if items:
            i = len(items)
            plural = " " if i == 1 else "s 1-"
            msg = f"strict_zip() argument {i+1} is shorter than argument{plural}{i}"
            raise ValueError(msg)

        sentinel = object()
        for i, iterator in enumerate(iterators[1:], 1):
            if next(iterator, sentinel) is not sentinel:
                plural = " " if i == 1 else "s 1-"
                msg = f"strict_zip() argument {i+1} is longer than argument{plural}{i}"
                raise ValueError(msg)
