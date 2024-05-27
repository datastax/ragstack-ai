import contextlib
import threading
from types import TracebackType
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple, Type

from cassandra.cluster import ResponseFuture, Session
from cassandra.query import PreparedStatement


class ConcurrentQueries(contextlib.AbstractContextManager):
    """Context manager for concurrent queries."""

    def __init__(self, session: Session, *, concurrency: int = 20) -> None:
        self._session = session
        self._semaphore = threading.Semaphore(concurrency)
        self._completion = threading.Condition()

        self._pending = 0

        self._error = None

    def _handle_result(
        self,
        result: Sequence[NamedTuple],
        future: ResponseFuture,
        callback: Optional[Callable[[Sequence[NamedTuple]], Any]],
    ):
        if callback is not None:
            callback(result)

        if future.has_more_pages:
            future.start_fetching_next_page()
        else:
            self._semaphore.release()
            with self._completion:
                self._pending -= 1
                if self._pending == 0:
                    self._completion.notify()

    def _handle_error(self, error):
        with self._completion:
            self._error = error
            self._completion.notify()

    def execute(
        self,
        query: PreparedStatement,
        parameters: Optional[Tuple] = None,
        callback: Optional[Callable[[Sequence[NamedTuple]], Any]] = None,
    ):
        with self._completion:
            self._pending += 1
            if self._error is not None:
                return

        self._semaphore.acquire()
        future: ResponseFuture = self._session.execute_async(query, parameters)
        future.add_callbacks(
            self._handle_result,
            self._handle_error,
            callback_kwargs={
                "future": future,
                "callback": callback,
            },
        )

    def __enter__(self) -> "ConcurrentQueries":
        return super().__enter__()

    def __exit__(
        self,
        _exc_type: Optional[Type[BaseException]],
        _exc_inst: Optional[BaseException],
        _exc_traceback: Optional[TracebackType],
    ) -> bool:
        with self._completion:
            while self._error is None and self._pending > 0:
                self._completion.wait()

        if self._error is not None:
            raise self._error

        # Don't swallow the exception.
        # We don't need to do anything with the exception (`_exc_*` parameters)
        # since returning false here will automatically re-raise it.
        return False
