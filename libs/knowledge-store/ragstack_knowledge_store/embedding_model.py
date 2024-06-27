from abc import ABC
from typing import List, Any, Optional
from collections import defaultdict

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
            
    def embed_mimes(self, texts: List[str], mime_types: List[str]) -> List[List[float]]:
        """Embed mime content."""

        # Extract main MIME types
        main_mime_types = [mime_type.split('/')[0] for mime_type in mime_types]

        # Group texts by main MIME types
        grouped_texts = defaultdict(list)
        index_mapping = defaultdict(list)
        for index, (text, main_mime_type) in enumerate(zip(texts, main_mime_types)):
            grouped_texts[main_mime_type].append(text)
            index_mapping[main_mime_type].append(index)

        # Initialize result list with None to preserve order
        embeddings = [None] * len(texts)

        # Process each MIME type group
        for mime_type, group_texts in grouped_texts.items():
            method_name = f"embed_{mime_type}s"
            if self.does_implement(method_name):
                # Bulk embedding method exists
                group_embeddings = self.invoke(method_name, group_texts)
                for idx, emb in zip(index_mapping[mime_type], group_embeddings):
                    embeddings[idx] = emb
            else:
                # No bulk method, fall back to individual methods
                singular_method_name = f"embed_{mime_type}"
                for text, idx in zip(group_texts, index_mapping[mime_type]):
                    if self.does_implement(singular_method_name):
                        embedding = self.invoke(singular_method_name, text)
                        embeddings[idx] = embedding
                    else:
                        raise NotImplementedError(f"No embedding method available for MIME type: {mime_type}, implemented methods: {self.implements()}.")

        # Ensure all embeddings are computed
        if None in embeddings:
            raise ValueError("Some embeddings were not computed correctly.")

        return embeddings
        
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

