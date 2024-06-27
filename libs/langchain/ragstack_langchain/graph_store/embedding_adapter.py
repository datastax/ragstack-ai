from typing import List
from ragstack_knowledge_store import EmbeddingModel

class EmbeddingAdapter(EmbeddingModel):
    def __init__(self, embeddings):
        super().__init__(embeddings,
                         method_map={'embed_texts': 'embed_documents',
                                     'aembed_texts': 'aembed_documents'})

