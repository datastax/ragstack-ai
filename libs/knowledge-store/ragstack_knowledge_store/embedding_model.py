from abc import ABC
from typing import List, Any, Optional

class EmbeddingModel(ABC):
    """Embedding model."""

    def __init__(self, embeddings: Any, method_map: Optional[dict] = None, other_methods: Optional[List[str]] = None):
        self.embeddings = embeddings
        self.method_name = {}
        method_map = method_map if method_map else {}
        other_methods = other_methods if other_methods else []

        base_methods = ['embed_texts', 'aembed_texts', 'embed_query', 'aembed_query']
        extended_methods = ['embed_images', 'aembed_images', 'embed_image', 'aembed_image']
        
        # Combining all method names, including those mapped
        all_methods = set(base_methods + extended_methods + other_methods + list(method_map.values()))

        for method in all_methods:
            mapped_method = method_map.get(method)
            if hasattr(embeddings, method):
                self.method_name[method] = method
            elif hasattr(embeddings, mapped_method) if mapped_method else False:
                self.method_name[method] = mapped_method
            else:
                self.method_name[method] = None

    def does_implement(self, method_name: str) -> bool:
        """Check if the method is implemented."""
        return self.method_name.get(method_name) is not None

    def implements(self) -> List[str]:
        """List of methods that are implemented"""
        return [method for method, impl in self.method_name.items() if impl is not None]

    def invoke(self, method_name: str, *args, **kwargs):
        """Invoke a synchronous method if it's implemented."""
        target_method = self.method_name.get(method_name)
        if target_method and hasattr(self.embeddings, target_method):
            return getattr(self.embeddings, target_method)(*args, **kwargs)
        else:
            raise NotImplementedError(f"{self.embeddings.__class__.__name__} does not implement {target_method}")

    async def ainvoke(self, method_name: str, *args, **kwargs):
        """Invoke an asynchronous method if it's implemented."""
        target_method = self.method_name.get(method_name)
        if target_method and hasattr(self.embeddings, target_method):
            return await getattr(self.embeddings, target_method)(*args, **kwargs)
        else:
            raise NotImplementedError(f"{self.embeddings.__class__.__name__} does not implement {target_method}")
        
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

