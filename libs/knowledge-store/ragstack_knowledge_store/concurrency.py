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
    from cassandra.query import PreparedStatement

logger = logging.getLogger(__name__)


class _Callback(Protocol):
    def __call__(self, rows: Sequence[Any], /) -> None: ...


class ConcurrentQueries(contextlib.AbstractContextManager["ConcurrentQueries"]):
    """Context manager for concurrent queries."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._completion = threading.Condition()
        self._pending = 0
        self._error: BaseException | None = None

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
            self._completion.notify()

    def execute(
        self,
        query: PreparedStatement,
        parameters: tuple[Any, ...] | None = None,
        callback: _Callback | None = None,
    ) -> None:
        """Execute a query concurrently.

        Because this is done concurrently, it expects a callback if you need
        to inspect the results.

        Args:
            query: The query to execute.
            parameters: Parameter tuple for the query. Defaults to `None`.
            callback: Callback to apply to the results. Defaults to `None`.
        """
        # TODO: We could have some form of throttling, where we track the number
        # of pending calls and queue things if it exceed some threshold.

        with self._completion:
            self._pending += 1
            if self._error is not None:
                return

        future: ResponseFuture = self._session.execute_async(query, parameters)
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
        # We don't need to do anything with the exception (`_exc_*` parameters)
        # since returning false here will automatically re-raise it.
        return False
