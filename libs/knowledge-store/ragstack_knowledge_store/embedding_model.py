from abc import ABC
from typing import List, Any, Optional

class EmbeddingModel(ABC):
    """Embedding model."""

    def __init__(self, embeddings:Any, other_methods:Optional[List[str]]=None):
        self.embeddings = embeddings
        self.method_cache = {}

        if other_methods is None:
            other_methods = []

        base_methods = ['embed_texts', 'aembed_texts', 'embed_query', 'aembed_query']
        extended_methods = ['embed_images', 'aembed_images', 'embed_image', 'aembed_image']

        for method_name in base_methods:
            self.method_cache[method_name] = True

        for method_name in extended_methods + other_methods:
            self.method_cache[method_name] = hasattr(embeddings, method_name)

    def does_implement(self, method_name: str) -> bool:
        """Check if the method is implemented."""
        return self.method_cache.get(method_name, False)

    def implements(self) -> List[str]:
        """List of methods that are implemented"""
        return [method for method, implemented in self.method_cache.items() if implemented]

    def invoke(self, method_name: str, *args, **kwargs):
        """Invoke a synchronous method if it's implemented."""
        if self.does_implement(method_name):
            return getattr(self.embeddings, method_name)(*args, **kwargs)
        else:
            raise NotImplementedError(f"{self.embeddings.__class__} does not implement {method_name}")

    async def ainvoke(self, method_name: str, *args, **kwargs):
        """Invoke an asynchronous method if it's implemented."""
        if self.does_implement(method_name):
            return await getattr(self.embeddings, method_name)(*args, **kwargs)
        else:
            raise NotImplementedError(f"{self.embeddings.__class__} does not implement {method_name}")
        
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts."""
        return self.invoke('embed_texts', texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.invoke('embed_query', text)

    async def aembed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts."""
        return await self.ainvoke('aembed_texts', texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return await self.ainvoke('aembed_query', text)

