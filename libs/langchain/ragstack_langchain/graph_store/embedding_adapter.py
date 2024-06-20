from typing import List
from ragstack_knowledge_store import EmbeddingModel

class EmbeddingAdapter(EmbeddingModel):
    def __init__(self, embeddings):
        super().__init__(embeddings, other_methods=['embed_documents,aembed_documents'])

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.invoke('embed_documents', texts)

    async def aembed_texts(self, texts: List[str]) -> List[List[float]]:
        return await self.ainvoke('aembed_documents', texts)
