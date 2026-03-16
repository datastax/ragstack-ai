from __future__ import annotations

import contextlib
import logging
import threading
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    NamedTuple,
    Protocol,
    Sequence,
)

if TYPE_CHECKING:
    from types import TracebackType

    from cassandra.cluster import ResponseFuture, Session
    from cassandra.query import PreparedStatement, SimpleStatement

logger = logging.getLogger(__name__)


class _Callback(Protocol):
    def __call__(self, rows: Sequence[Any], /) -> None:
        ...


class ConcurrentQueries(contextlib.AbstractContextManager["ConcurrentQueries"]):
    """Context manager for concurrent queries with a max limit of 5 ongoing queries."""

    _MAX_CONCURRENT_QUERIES = 5

    def __init__(self, session: Session) -> None:
        self._session = session
        self._completion = threading.Condition()
        self._pending = 0
        self._error: BaseException | None = None
        self._semaphore = threading.Semaphore(self._MAX_CONCURRENT_QUERIES)

    def _handle_result(
        self,
        result: Sequence[NamedTuple],
        future: ResponseFuture,
        callback: Callable[[Sequence[NamedTuple]], Any] | None,
    ) -> None:
        if callback is not None:
            callback(result)

        if future.has_more_pages:
            future.start_fetching_next_page()
        else:
            with self._completion:
                self._pending -= 1
                self._semaphore.release()  # Release the semaphore once a query completes
                if self._pending == 0:
                    self._completion.notify()

    def _handle_error(self, error: BaseException, future: ResponseFuture) -> None:
        logger.error(
            "Error executing query: %s",
            future.query,
            exc_info=error,
        )
        with self._completion:
            self._error = error
            self._pending -= 1  # Decrement pending count
            self._semaphore.release()  # Release the semaphore on error
            self._completion.notify()

    def execute(
        self,
        query: PreparedStatement | SimpleStatement,
        parameters: tuple[Any, ...] | None = None,
        callback: _Callback | None = None,
        timeout: float | None = None,
    ) -> None:
        """Execute a query concurrently with a max of 5 concurrent queries.

        Args:
            query: The query to execute.
            parameters: Parameter tuple for the query. Defaults to `None`.
            callback: Callback to apply to the results. Defaults to `None`.
            timeout: Timeout to use (if not the session default).
        """
        with self._completion:
            if self._error is not None:
                return

        # Acquire the semaphore before proceeding to ensure we do not exceed the max limit
        self._semaphore.acquire()

        with self._completion:
            if self._error is not None:
                # Release semaphore before returning
                self._semaphore.release()
                return
            self._pending += 1

        try:
            execute_kwargs = {}
            if timeout is not None:
                execute_kwargs["timeout"] = timeout
            future: ResponseFuture = self._session.execute_async(
                query,
                parameters,
                **execute_kwargs,
            )
            future.add_callbacks(
                self._handle_result,
                self._handle_error,
                callback_kwargs={
                    "future": future,
                    "callback": callback,
                },
                errback_kwargs={
                    "future": future,
                },
            )
        except Exception as e:
            with self._completion:
                self._error = e
                self._pending -= 1  # Decrement pending count
                self._semaphore.release()  # Release semaphore
                self._completion.notify()
            raise

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_inst: BaseException | None,
        _exc_traceback: TracebackType | None,
    ) -> Literal[False]:
        with self._completion:
            while self._error is None and self._pending > 0:
                self._completion.wait()

        if self._error is not None:
            raise self._error

        # Don't swallow the exception.
        return False
