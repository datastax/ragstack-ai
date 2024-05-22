import json
from itertools import repeat
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union, cast

from cassandra.cluster import ResponseFuture, Session
from cassandra.query import BatchStatement
from cassio.config import check_resolve_keyspace, check_resolve_session
from langchain_core.embeddings import Embeddings

from .traverse import Node, Relation, atraverse, traverse
from .utils import batched


def _serialize_md_dict(md_dict: Dict[str, Any]) -> str:
    return json.dumps(md_dict, separators=(",", ":"), sort_keys=True)


def _deserialize_md_dict(md_string: str) -> Dict[str, Any]:
    return cast(Dict[str, Any], json.loads(md_string))


def _parse_node(row) -> Node:
    return Node(
        name=row.name,
        type=row.type,
        properties=_deserialize_md_dict(row.properties_json) if row.properties_json else dict(),
    )


class CassandraKnowledgeGraph:
    def __init__(
        self,
        node_table: str = "entities",
        edge_table: str = "relationships",
        text_embeddings: Optional[Embeddings] = None,
        session: Optional[Session] = None,
        keyspace: Optional[str] = None,
        apply_schema: bool = True,
    ) -> None:
        """
        Create a Cassandra Knowledge Graph.

        Parameters:
        - node_table: Name of the table containing nodes. Defaults to `"entities"`.
        - edge_table: Name of the table containing edges. Defaults to `"relationships"`.
        _ text_embeddings: Name of the embeddings to use, if any.
        - session: The Cassandra `Session` to use. If not specified, uses the default `cassio`
          session, which requires `cassio.init` has been called.
        - keyspace: The Cassandra keyspace to use. If not specified, uses the default `cassio`
          keyspace, which requires `cassio.init` has been called.
        - apply_schema: If true, the node table and edge table are created.
        """

        session = check_resolve_session(session)
        keyspace = check_resolve_keyspace(keyspace)

        self._text_embeddings = text_embeddings
        self._text_embeddings_dim = (
            # Embedding vectors must have dimension:
            #  > 0 to be created at all.
            #  > 1 to support cosine distance.
            # So we default to 2.
            len(text_embeddings.embed_query("test string")) if text_embeddings else 2
        )

        self._session = session
        self._keyspace = keyspace

        self._node_table = node_table
        self._edge_table = edge_table

        if apply_schema:
            self._apply_schema()

        self._insert_node = self._session.prepare(
            f"""INSERT INTO {keyspace}.{node_table} (
                    name, type, text_embedding, properties_json
                ) VALUES (?, ?, ?, ?)
            """
        )

        self._insert_relationship = self._session.prepare(
            f"""
            INSERT INTO {keyspace}.{edge_table} (
                source_name, source_type, target_name, target_type, edge_type
            ) VALUES (?, ?, ?, ?, ?)
            """
        )

        self._query_relationship = self._session.prepare(
            f"""
            SELECT name, type, properties_json
            FROM {keyspace}.{node_table}
            WHERE name = ? AND type = ?
            """
        )

        self._query_nodes_by_embedding = self._session.prepare(
            f"""
            SELECT name, type, properties_json
            FROM {keyspace}.{node_table}
            ORDER BY text_embedding ANN OF ?
            LIMIT ?
            """
        )

    def _apply_schema(self):
        # Partition by `name` and cluster by `type`.
        # Each `(name, type)` pair is a unique node.
        # We can enumerate all `type` values for a given `name` to identify ambiguous terms.
        self._session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._keyspace}.{self._node_table} (
                name TEXT,
                type TEXT,
                properties_json TEXT,
                text_embedding VECTOR<FLOAT, {self._text_embeddings_dim}>,
                PRIMARY KEY (name, type)
            );
            """
        )

        self._session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._keyspace}.{self._edge_table} (
                source_name TEXT,
                source_type TEXT,
                target_name TEXT,
                target_type TEXT,
                edge_type TEXT,
                PRIMARY KEY ((source_name, source_type), target_name, target_type, edge_type)
            );
            """
        )

        self._session.execute(
            f"""
            CREATE CUSTOM INDEX IF NOT EXISTS {self._node_table}_text_embedding_index
            ON {self._keyspace}.{self._node_table} (text_embedding)
            USING 'StorageAttachedIndex';
            """
        )

        self._session.execute(
            f"""
            CREATE CUSTOM INDEX IF NOT EXISTS {self._edge_table}_type_index
            ON {self._keyspace}.{self._edge_table} (edge_type)
            USING 'StorageAttachedIndex';
            """
        )

    def _send_query_nearest_node(self, node: str, k: int = 1) -> ResponseFuture:
        return self._session.execute_async(
            self._query_nodes_by_embedding,
            (
                self._text_embeddings.embed_query(node),
                k,
            ),
        )

    # TODO: Allow filtering by node predicates and/or minimum similarity.
    def query_nearest_nodes(self, nodes: Iterable[str], k: int = 1) -> Iterable[Node]:
        """
        For each node, return the nearest nodes in the table.

        Parameters:
        - nodes: The strings to search for in the list of nodes.
        - k: The number of similar nodes to retrieve for each string.
        """
        if self._text_embeddings is None:
            raise ValueError("Unable to query for nearest nodes without embeddings")

        node_futures: Iterable[ResponseFuture] = [
            self._send_query_nearest_node(n, k) for n in nodes
        ]

        nodes = {_parse_node(n) for node_future in node_futures for n in node_future.result()}
        return list(nodes)

    # TODO: Introduce `ainsert` for async insertions.
    def insert(
        self,
        elements: Iterable[Union[Node, Relation]],
    ) -> None:
        for batch in batched(elements, n=4):
            from yaml import dump
            text_embeddings = (
                iter(
                    self._text_embeddings.embed_documents(
                        [dump(n) for n in batch if isinstance(n, Node)]
                    )
                )
                if self._text_embeddings
                else repeat([0.0, 1.0])
            )

            batch_statement = BatchStatement()
            for element in batch:
                if isinstance(element, Node):
                    properties_json = _serialize_md_dict(element.properties)
                    batch_statement.add(
                        self._insert_node,
                        (element.name, element.type, next(text_embeddings), properties_json),
                    )
                elif isinstance(element, Relation):
                    batch_statement.add(
                        self._insert_relationship,
                        (
                            element.source.name,
                            element.source.type,
                            element.target.name,
                            element.target.type,
                            element.type,
                        ),
                    )
                else:
                    raise ValueError(f"Unsupported element type: {element}")

            # TODO: Support concurrent execution of these statements.
            self._session.execute(batch_statement)

    def subgraph(
        self,
        start: Node | Sequence[Node],
        edge_filters: Sequence[str] = (),
        steps: int = 3,
    ) -> Tuple[Iterable[Node], Iterable[Relation]]:
        """
        Retrieve the sub-graph from the given starting nodes.
        """
        edges = self.traverse(start, edge_filters, steps)

        # Create the set of nodes.
        nodes = {n for e in edges for n in (e.source, e.target)}

        # Retrieve the set of nodes to get the properties.

        # TODO: We really should have a NodeKey separate from Node. Otherwise, we end
        # up in a state where two nodes can be the "same" but with different properties,
        # etc.

        node_futures: Iterable[ResponseFuture] = [
            self._session.execute_async(self._query_relationship, (n.name, n.type)) for n in nodes
        ]

        nodes = [_parse_node(n) for future in node_futures for n in future.result()]

        return (nodes, edges)

    def traverse(
        self,
        start: Node | Sequence[Node],
        edge_filters: Sequence[str] = (),
        steps: int = 3,
    ) -> Iterable[Relation]:
        """
        Traverse the graph from the given starting nodes and return the resulting sub-graph.

        Parameters:
        - start: The starting node or nodes.
        - edge_filters: Filters to apply to the edges being traversed.
        - steps: The number of steps of edges to follow from a start node.

        Returns:
        An iterable over relations in the traversed sub-graph.
        """
        return traverse(
            start=start,
            edge_table=self._edge_table,
            edge_source_name="source_name",
            edge_source_type="source_type",
            edge_target_name="target_name",
            edge_target_type="target_type",
            edge_type="edge_type",
            edge_filters=edge_filters,
            steps=steps,
            session=self._session,
            keyspace=self._keyspace,
        )

    async def atraverse(
        self,
        start: Node | Sequence[Node],
        edge_filters: Sequence[str] = (),
        steps: int = 3,
    ) -> Iterable[Relation]:
        """
        Traverse the graph from the given starting nodes and return the resulting sub-graph.

        Parameters:
        - start: The starting node or nodes.
        - edge_filters: Filters to apply to the edges being traversed.
        - steps: The number of steps of edges to follow from a start node.

        Returns:
        An iterable over relations in the traversed sub-graph.
        """
        return await atraverse(
            start=start,
            edge_table=self._edge_table,
            edge_source_name="source_name",
            edge_source_type="source_type",
            edge_target_name="target_name",
            edge_target_type="target_type",
            edge_type="edge_type",
            edge_filters=edge_filters,
            steps=steps,
            session=self._session,
            keyspace=self._keyspace,
        )
