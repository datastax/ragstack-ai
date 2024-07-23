"""
Normalization of metadata policy specification options
"""

from ragstack_knowledge_store.graph_store import GraphStore, MetadataIndexingMode


class TestNormalizeMetadataPolicy:
    def test_normalize_metadata_policy(self) -> None:
        mdp1 = GraphStore._normalize_metadata_indexing_policy("all")  # noqa: SLF001
        assert mdp1 == (MetadataIndexingMode.DEFAULT_TO_SEARCHABLE, set())
        mdp2 = GraphStore._normalize_metadata_indexing_policy("none")  # noqa: SLF001
        assert mdp2 == (MetadataIndexingMode.DEFAULT_TO_UNSEARCHABLE, set())
        mdp3 = GraphStore._normalize_metadata_indexing_policy(  # noqa: SLF001
            ("default_to_Unsearchable", ["x", "y"]),
        )
        assert mdp3 == (MetadataIndexingMode.DEFAULT_TO_UNSEARCHABLE, {"x", "y"})
        mdp4 = GraphStore._normalize_metadata_indexing_policy(  # noqa: SLF001
            ("DenyList", ["z"]),
        )
        assert mdp4 == (MetadataIndexingMode.DEFAULT_TO_SEARCHABLE, {"z"})
        # s
        mdp5 = GraphStore._normalize_metadata_indexing_policy(  # noqa: SLF001
            ("deny_LIST", "singlefield")
        )
        assert mdp5 == (MetadataIndexingMode.DEFAULT_TO_SEARCHABLE, {"singlefield"})
