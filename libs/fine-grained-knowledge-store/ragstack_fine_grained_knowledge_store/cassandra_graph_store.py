from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from cassandra.cluster import Session
from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs.graph_document import Node as LangChainNode
from langchain_community.graphs.graph_store import GraphStore
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable, RunnableLambda

from knowledge_graph.knowledge_graph import CassandraKnowledgeGraph

from .traverse import Node, Relation


def _elements(documents: Iterable[GraphDocument]) -> Iterable[Union[Node, Relation]]:
    def _node(node: LangChainNode) -> Node:
        return Node(name=str(node.id), type=node.type)

    for document in documents:
        for node in document.nodes:
            yield _node(node)
        for edge in document.relationships:
            yield Relation(source=_node(edge.source), target=_node(edge.target), type=edge.type)


class CassandraGraphStore(GraphStore):
    def __init__(
        self,
        node_table: str = "entities",
        edge_table: str = "relationships",
        text_embeddings: Optional[Embeddings] = None,
        session: Optional[Session] = None,
        keyspace: Optional[str] = None,
    ) -> None:
        """
        Create a Cassandra Graph Store.

        Before calling this, you must initialize cassio with `cassio.init`, or
        provide valid session and keyspace values.
        """
        self.graph = CassandraKnowledgeGraph(
            node_table=node_table,
            edge_table=edge_table,
            text_embeddings=text_embeddings,
            session=session,
            keyspace=keyspace,
        )

    def add_graph_documents(
        self, graph_documents: List[GraphDocument], include_source: bool = False
    ) -> None:
        # TODO: Include source.
        self.graph.insert(_elements(graph_documents))

    # TODO: should this include the types of each node?
    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        raise ValueError("Querying Cassandra should use `as_runnable`.")

    def as_runnable(self, steps: int = 3, edge_filters: Sequence[str] = []) -> Runnable:
        """
        Return a runnable that retrieves the sub-graph near the input entity or entities.

        Parameters:
        - steps: The maximum distance to follow from the starting points.
        - edge_filters: Predicates to use for filtering the edges.
        """
        return RunnableLambda(func=self.graph.traverse, afunc=self.graph.atraverse).bind(
            steps=steps,
            edge_filters=edge_filters,
        )
