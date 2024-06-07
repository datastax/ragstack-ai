from ragstack_knowledge_store.explicit_edge_extractor import ExplicitEdgeExtractor


class ParentEdgeExtractor(ExplicitEdgeExtractor):
    def __init__(self, parent_field: str = "parent_content_id") -> None:
        """Extract an edge from a node to its parent.

        NOTE: If no parent is specified, no edge will be created. If an edge is
        specified and the target node doesn't exist, an error will be raised.
        This means the target must be added in the same or an earlier batch of
        nodes.

        Args:
            parent_field: The metadata field containing the parent content ID.
        """
        super().__init__(edges_field=parent_field, kind="has_parent", bidir=False)
