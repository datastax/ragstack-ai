"""
Defines constants used across the system for configuring and operating the ColBERT model for semantic
search and retrieval tasks.

Constants:
- DEFAULT_COLBERT_MODEL: Specifies the hugging-face identifier of the default ColBERT model to be used.
- DEFAULT_COLBERT_DIM: Sets the default dimension for embeddings produced by the ColBERT model.
- MAX_MODEL_TOKENS: The maximum number of tokens that can be processed by the model in a single input.
- CHUNK_MAX_PER_DOC: The upper limit on the number of text chunks that a single document can be divided into.
"""

DEFAULT_COLBERT_MODEL = "colbert-ir/colbertv2.0"

DEFAULT_COLBERT_DIM = 128

MAX_MODEL_TOKENS = 512
