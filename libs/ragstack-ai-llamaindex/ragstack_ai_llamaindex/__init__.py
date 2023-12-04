from importlib import metadata

import llama_index

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)


# LlamaHub is disabled while using RAGStack AI, because it would download unknown software from the Internet.
def disabled_download_loader(*args, **kwargs):
    raise ImportError(
        "LlamaHub is disabled while using RAGStack AI, because it would download unknown software from the Internet.")


llama_index.download_loader = disabled_download_loader
