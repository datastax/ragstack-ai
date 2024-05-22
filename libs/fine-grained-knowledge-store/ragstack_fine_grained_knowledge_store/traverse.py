import asyncio
import threading
from typing import Any, Dict, Iterable, NamedTuple, Optional, Sequence

from cassandra.cluster import PreparedStatement, ResponseFuture, Session
from cassio.config import check_resolve_keyspace, check_resolve_session


class Node(NamedTuple):
    name: str
    type: str
    properties: Dict[str, Any] = {}

    def __repr__(self):
        return f"{self.name} ({self.type})"

    def __hash__(self):
        return hash(self.name) * hash(self.type)

    def __eq__(self, value) -> bool:
        if not isinstance(value, Node):
            return False

        return self.name == value.name and self.type == value.type


class Relation(NamedTuple):
    source: Node
    target: Node
    type: str

    def __repr__(self):
        return f"{self.source} -> {self.target}: {self.type}"


def _parse_relation(row) -> Relation:
    return Relation(
        source=Node(name=row.source_name, type=row.source_type),
        target=Node(name=row.target_name, type=row.target_type),
        type=row.type,
    )


def _prepare_edge_query(
    edge_table: str,
    edge_source_name: str,
    edge_source_type: str,
    edge_target_name: str,
    edge_target_type: str,
    edge_type: str,
    edge_filters: Sequence[str],
    session: Session,
    keyspace: str,
) -> PreparedStatement:
    """Return the query for the edges from a given source."""
    query = f"""
        SELECT
            {edge_source_name} AS source_name,
            {edge_source_type} AS source_type,
            {edge_target_name} AS target_name,
            {edge_target_type} AS target_type,
            {edge_type} AS type
        FROM {keyspace}.{edge_table}
        WHERE {edge_source_name} = ?
        AND {edge_source_type} = ?"""
    if edge_filters:
        query = "\n        AND ".join([query] + edge_filters)
    return session.prepare(query)


def traverse(
    start: Node | Sequence[Node],
    edge_table: str,
    edge_source_name: str = "source_name",
    edge_source_type: str = "source_type",
    edge_target_name: str = "target_name",
    edge_target_type: str = "target_type",
    edge_type: str = "edge_type",
    edge_filters: Sequence[str] = (),
    steps: int = 3,
    session: Optional[Session] = None,
    keyspace: Optional[str] = None,
) -> Iterable[Relation]:
    """
    Traverse the graph from the given starting nodes and return the resulting sub-graph.

    Parameters:
    - start: The starting node or nodes.
    - edge_table: The table containing the edges.
    - edge_source_name: The name of the column containing edge source names.
    - edge_source_type: The name of the column containing edge source types.
    - edge_target_name: The name of the column containing edge target names.
    - edge_target_type: The name of the column containing edge target types.
    - edge_type: The name of the column containing edge types.
    - edge_filters: Filters to apply to the edges being traversed.
    - steps: The number of steps of edges to follow from a start node.
    - session: The session to use for executing the query. If not specified,
      it will use th default cassio session.
    - keyspace: The keyspace to use for the query. If not specified, it will
      use the default cassio keyspace.

    Returns:
    An iterable over relations in the traversed sub-graph.
    """
    if len(start) == 0:
        return []

    session = check_resolve_session(session)
    keyspace = check_resolve_keyspace(keyspace)

    pending = set()
    distances = {}
    results = set()
    query = _prepare_edge_query(
        edge_table=edge_table,
        edge_source_name=edge_source_name,
        edge_source_type=edge_source_type,
        edge_target_name=edge_target_name,
        edge_target_type=edge_target_type,
        edge_type=edge_type,
        edge_filters=edge_filters,
        session=session,
        keyspace=keyspace,
    )

    condition = threading.Condition()
    error = None

    def handle_result(rows, source_distance: int, request: ResponseFuture):
        relations = map(_parse_relation, rows)
        with condition:
            if source_distance < steps:
                for r in relations:
                    results.add(r)
                    fetch_relationships(source_distance + 1, r.target)
            else:
                results.update(relations)

        if request.has_more_pages:
            request.start_fetching_next_page()
        else:
            with condition:
                if request._req_id in pending:
                    pending.remove(request._req_id)
                if len(pending) == 0:
                    condition.notify()

    def handle_error(e):
        nonlocal error
        with condition:
            error = e
            condition.notify()

    def fetch_relationships(distance: int, source: Node) -> None:
        """
        Fetch relationships from node `source` is found at `distance`.

        This will retrieve the edges from `source`, and visit the resulting
        nodes at distance `distance + 1`.
        """
        with condition:
            old_distance = distances.get(source)
            if old_distance is not None and old_distance <= distance:
                # Already discovered at that distance.
                return

            distances[source] = distance

            request: ResponseFuture = session.execute_async(query, (source.name, source.type))
            pending.add(request._req_id)
            request.add_callbacks(
                handle_result,
                handle_error,
                callback_kwargs={"source_distance": distance, "request": request},
            )

    with condition:
        if isinstance(start, Node):
            start = [start]
        for source in start:
            fetch_relationships(1, source)

        condition.wait()

        if error is not None:
            raise error
        else:
            return results


class AsyncPagedQuery(object):
    def __init__(self, depth: int, response_future: ResponseFuture):
        self.loop = asyncio.get_running_loop()
        self.depth = depth
        self.response_future = response_future
        self.current_page_future = asyncio.Future()
        self.response_future.add_callbacks(self._handle_page, self._handle_error)

    def _handle_page(self, rows):
        self.loop.call_soon_threadsafe(self.current_page_future.set_result, rows)

    def _handle_error(self, error):
        self.loop.call_soon_threadsafe(self.current_page_future.set_exception, error)

    async def next(self):
        page = [_parse_relation(r) for r in await self.current_page_future]

        if self.response_future.has_more_pages:
            self.current_page_future = asyncio.Future()
            self.response_future.start_fetching_next_page()
            return (self.depth, page, self)
        else:
            return (self.depth, page, None)


async def atraverse(
    start: Node | Sequence[Node],
    edge_table: str,
    edge_source_name: str = "source_name",
    edge_source_type: str = "source_type",
    edge_target_name: str = "target_name",
    edge_target_type: str = "target_type",
    edge_type: str = "edge_type",
    edge_filters: Sequence[str] = [],
    steps: int = 3,
    session: Optional[Session] = None,
    keyspace: Optional[str] = None,
) -> Iterable[Relation]:
    """
    Async traversal of the graph from the given starting nodes and return the resulting sub-graph.

    Parameters:
    - start: The starting node or nodes.
    - edge_table: The table containing the edges.
    - edge_source_name: The name of the column containing edge source names.
    - edge_source_type: The name of the column containing edge source types.
    - edge_target_name: The name of the column containing edge target names.
    - edge_target_type: The name of the column containing edge target types.
    - edge_type: The name of the column containing edge types.
    - edge_filters: Filters to apply to the edges being traversed.
      Currently, this is specified as a dictionary containing the name
      of the edge field to filter on and the CQL predicate to apply.
      For example `{"foo": "IN ['a', 'b', 'c']"}`.
    - steps: The number of steps of edges to follow from a start node.
    - session: The session to use for executing the query. If not specified,
      it will use th default cassio session.
    - keyspace: The keyspace to use for the query. If not specified, it will
      use the default cassio keyspace.

    Returns:
    An iterable over relations in the traversed sub-graph.
    """

    session = check_resolve_session(session)
    keyspace = check_resolve_keyspace(keyspace)

    # Prepare the query.
    #
    # We reprepare this for each traversal since each call may have different
    # filters.
    #
    # TODO: We should cache this at least for the common case of no-filters.
    query = _prepare_edge_query(
        edge_table=edge_table,
        edge_source_name=edge_source_name,
        edge_source_type=edge_source_type,
        edge_target_name=edge_target_name,
        edge_target_type=edge_target_type,
        edge_type=edge_type,
        edge_filters=edge_filters,
        session=session,
        keyspace=keyspace,
    )

    def fetch_relation(tg: asyncio.TaskGroup, depth: int, source: Node) -> AsyncPagedQuery:
        paged_query = AsyncPagedQuery(
            depth, session.execute_async(query, (source.name, source.type))
        )
        return tg.create_task(paged_query.next())

    results = set()
    async with asyncio.TaskGroup() as tg:
        if isinstance(start, Node):
            start = [start]

        discovered = {t: 0 for t in start}
        pending = {fetch_relation(tg, 1, source) for source in start}

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for future in done:
                depth, relations, more = future.result()
                for relation in relations:
                    results.add(relation)

                # Schedule the future for more results from the same query.
                if more is not None:
                    pending.add(tg.create_task(more.next()))

                # Schedule futures for the next step.
                if depth < steps:
                    # We've found a path of length `depth` to each of the targets.
                    # We need to update `discovered` to include the shortest path.
                    # And build `to_visit` to be all of the targets for which this is
                    # the new shortest path.
                    to_visit = set()
                    for r in relations:
                        previous = discovered.get(r.target, steps + 1)
                        if depth < previous:
                            discovered[r.target] = depth
                            to_visit.add(r.target)

                    for source in to_visit:
                        pending.add(fetch_relation(tg, depth + 1, source))

    return results
