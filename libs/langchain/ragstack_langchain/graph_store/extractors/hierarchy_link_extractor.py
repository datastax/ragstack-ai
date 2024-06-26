from typing import List, Set, TypeAlias

from ragstack_langchain.graph_store.extractors.link_extractor import LinkExtractor
from ragstack_langchain.graph_store.links import Link


HierarchyInput: TypeAlias = List[str]

class HierarchyLinkExtractor(LinkExtractor[HierarchyInput]):
    def __init__(self,
                 kind: str = "hierarchy",
                 up_links: bool = True,
                 down_links: bool = False,
                 sibling_links: bool = False):
        """Extract links from a document hierarchy.

        Args:
            kind: Kind of links to produce with this extractor.
            up_links: Link from a section to it's parent.
            down_links: Link from a section to it's children.
            sibling_links: Link from a section to other sections with the same parent.
        """
        self._kind = kind
        self._up_links = up_links
        self._down_links = down_links
        self._sibling_links = sibling_links

    def extract_one(
            self,
            input: HierarchyInput,
    ) -> Set[Link]:
        this_path = "/".join(input)
        parent_path = None

        links = set()
        if self._up_links:
            links.add(Link.incoming(kind=self._kind, tag=f"up:{this_path}"))
        if self._down_links:
            links.add(Link.outgoing(kind=self._kind, tag=f"down:{this_path}"))

        if len(input) >= 1:
            parent_path = "/".join(input[0:-1])
            if self._up_links and len(input) > 1:
                links.add(Link.outgoing(kind=self._kind, tag=f"up:{parent_path}"))
            if self._down_links and len(input) > 1:
                links.add(Link.incoming(kind=self._kind, tag=f"down:{parent_path}"))
            if self._sibling_links:
                links.add(Link.bidir(kind=self._kind, tag=f"sib:{parent_path}"))

        return links