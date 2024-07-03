import logging
from typing import List

import torch
from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.infra.config import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from ragstack_colbert import ColbertEmbeddingModel, Embedding
from ragstack_colbert.constant import DEFAULT_COLBERT_MODEL
from torch import Tensor
from torch.nn.functional import cosine_similarity

from .baseline_tensors import baseline_tensors

"""
These tests are for embedding drift and evaluate if the embeddings are changing over time due to
1) changes in the model or drift in model weights
   model drift does not necessarily mean the test fails, but it is a signal that the model is changing
2) changes in the embedding implementation
   if the implementation changes, we need to investigate if the change is intentional or not
"""  # noqa: E501

# 8 chunks with 220 tokens each for testing
# the above 645 per token embeddings matches arctic_botany_chunks's token embedding

arctic_botany_dict = {
    "Introduction to Arctic Botany": "Arctic botany is the study of plant life in the Arctic, a region characterized by extreme cold, permafrost, and minimal sunlight for much of the year. Despite these harsh conditions, a diverse range of flora thrives here, adapted to survive with minimal water, low temperatures, and high light levels during the summer. This introduction aims to shed light on the resilience and adaptation of Arctic plants, setting the stage for a deeper dive into the unique botanical ecosystem of the Arctic.",  # noqa: E501
    "Arctic Plant Adaptations": "Plants in the Arctic have developed unique adaptations to endure the extreme climate. Perennial growth, antifreeze proteins, and a short growth cycle are among the evolutionary solutions. These adaptations not only allow the plants to survive but also to reproduce in short summer months. Arctic plants often have small, dark leaves to absorb maximum sunlight, and some species grow in cushion or mat forms to resist cold winds. Understanding these adaptations provides insights into the resilience of Arctic flora.",  # noqa: E501
    "The Tundra Biome": "The Arctic tundra is a vast, treeless biome where the subsoil is permanently frozen. Here, the vegetation is predominantly composed of dwarf shrubs, grasses, mosses, and lichens. The tundra supports a surprisingly rich biodiversity, adapted to its cold, dry, and windy conditions. The biome plays a crucial role in the Earth's climate system, acting as a carbon sink. However, it's sensitive to climate change, with thawing permafrost and shifting vegetation patterns.",  # noqa: E501
    "Arctic Plant Biodiversity": "Despite the challenging environment, the Arctic boasts a significant variety of plant species, each adapted to its niche. From the colorful blooms of Arctic poppies to the hardy dwarf willows, these plants form a complex ecosystem. The biodiversity of Arctic flora is vital for local wildlife, providing food and habitat. This diversity also has implications for Arctic peoples, who depend on certain plant species for food, medicine, and materials.",  # noqa: E501
    "Climate Change and Arctic Flora": "Climate change poses a significant threat to Arctic botany, with rising temperatures, melting permafrost, and changing precipitation patterns. These changes can lead to shifts in plant distribution, phenology, and the composition of the Arctic flora. Some species may thrive, while others could face extinction. This dynamic is critical to understanding future Arctic ecosystems and their global impact, including feedback loops that may exacerbate global warming.",  # noqa: E501
    "Research and Conservation in the Arctic": "Research in Arctic botany is crucial for understanding the intricate balance of this ecosystem and the impacts of climate change. Scientists conduct studies on plant physiology, genetics, and ecosystem dynamics. Conservation efforts are focused on protecting the Arctic's unique biodiversity through protected areas, sustainable management practices, and international cooperation. These efforts aim to preserve the Arctic flora for future generations and maintain its role in the global climate system.",  # noqa: E501
    "Traditional Knowledge and Arctic Botany": "Indigenous peoples of the Arctic have a deep connection with the land and its plant life. Traditional knowledge, passed down through generations, includes the uses of plants for nutrition, healing, and materials. This body of knowledge is invaluable for both conservation and understanding the ecological relationships in Arctic ecosystems. Integrating traditional knowledge with scientific research enriches our comprehension of Arctic botany and enhances conservation strategies.",  # noqa: E501
    "Future Directions in Arctic Botanical Studies": "The future of Arctic botany lies in interdisciplinary research, combining traditional knowledge with modern scientific techniques. As the Arctic undergoes rapid changes, understanding the ecological, cultural, and climatic dimensions of Arctic flora becomes increasingly important. Future research will need to address the challenges of climate change, explore the potential for Arctic plants in biotechnology, and continue to conserve this unique biome. The resilience of Arctic flora offers lessons in adaptation and survival relevant to global challenges.",  # noqa: E501
}

arctic_botany_chunks = list(arctic_botany_dict.values())


# a uility function to evaluate similarity of two embeddings at per token level
def are_they_similar(embedded_chunks: List[Embedding], tensors: List[Tensor]):
    n = 0
    pdist = torch.nn.PairwiseDistance(p=2)
    for embedding in embedded_chunks:
        for vector in embedding:
            vector_tensor = torch.tensor(vector)
            assert vector_tensor.shape == tensors[n].shape

            # we still have outlier over the specified limit but almost 0
            assert pdist(vector_tensor, tensors[n]).item() < 0.0001

            similarity = cosine_similarity(
                vector_tensor.unsqueeze(0), tensors[n].unsqueeze(0)
            )
            assert similarity.item() > 0.999
            n = n + 1

    assert n == len(tensors)


def test_embeddings_with_baseline():
    colbert = ColbertEmbeddingModel(
        doc_maxlen=220,
        nbits=2,
        kmeans_niters=4,
        checkpoint=DEFAULT_COLBERT_MODEL,
    )

    """
    1. test any drift from the baseline
    2. test any deviation from the embedding functions

    since we don't have a graph or storage to keep track any degreation,
    please add to the model and implementions resultsed euclidian and cosine threshold change
    2024-04-08 default model - https://huggingface.co/colbert-ir/colbertv2.0
    """  # noqa: E501
    embeddings: List[Embedding] = colbert.embed_texts(arctic_botany_chunks)

    pdist = torch.nn.PairwiseDistance(p=2)
    embedded_tensors = []
    n = 0
    for embedding in embeddings:
        for vector in embedding:
            vector_tensor = torch.tensor(vector)
            embedded_tensors.append(vector_tensor)
            distance = torch.norm(vector_tensor - baseline_tensors[n])
            assert abs(distance) < 0.001
            # another way to measure pairwise distance
            # it must be a positive since it's from square root
            assert pdist(vector_tensor, baseline_tensors[n]).item() < 0.001

            similarity = cosine_similarity(
                vector_tensor.unsqueeze(0), baseline_tensors[n].unsqueeze(0)
            )
            assert similarity.shape == torch.Size([1])  # this has to be scalar
            # debug code to identify which token deviates
            if similarity.item() < 0.99:
                logging.warning(f"n = {n}, similarity = {similarity.item()}")
            assert similarity.item() > 0.99
            n = n + 1

    assert len(embedded_tensors) == 645

    # test against the same function to make sure to generate the same embeddings.
    # use the same ColBERT configurations but reload the checkpoint with the default
    # settings.
    # this also make sure the default ColBERT configurations have not changed.
    colbert2 = ColbertEmbeddingModel(
        checkpoint=DEFAULT_COLBERT_MODEL,
    )
    embedded_chunks2 = colbert2.embed_texts(arctic_botany_chunks)

    are_they_similar(embedded_chunks2, embedded_tensors)


def test_colbert_embedding_against_vanilla_impl():
    # this is a vanilla ColBERT embedding in a list of per token embeddings
    # based on the just Stanford ColBERT library
    cf = ColBERTConfig(checkpoint="colbert-ir/colbertv2.0")
    cp = Checkpoint(cf.checkpoint, colbert_config=cf)
    encoder = CollectionEncoder(cf, cp)

    embeddings_flat, _ = encoder.encode_passages(arctic_botany_chunks)

    colbert_svc = ColbertEmbeddingModel(
        checkpoint=DEFAULT_COLBERT_MODEL,
    )
    embedded_chunks = colbert_svc.embed_texts(arctic_botany_chunks)

    are_they_similar(embedded_chunks, embeddings_flat)


def model_embedding(model: str):
    logging.info(f"test model compatibility {model}")
    colbert_svc = ColbertEmbeddingModel(
        checkpoint=model,
        query_maxlen=32,
    )
    embeddings = colbert_svc.embed_texts(arctic_botany_chunks)

    assert len(embeddings) == 8
    n = 0
    for embedding in embeddings:
        for vector in embedding:
            assert len(vector) == 128
            n = n + 1

    assert n == 645

    # recall embeddings test
    embedding = colbert_svc.embed_query(
        query="What adaptations enable Arctic plants to survive and thrive "
        "in extremely cold temperatures and minimal sunlight?",
        query_maxlen=32,
    )
    assert len(embedding) == 32


def test_compatible_models():
    # ColBERT models and Google BERT models on HF
    # test representive models's compatibility with this repo's ColBERT embedding
    # evaluation is not within this test scope
    models = [
        "colbert-ir/colbertv1.9",
        # "colbert-ir/colbertv2.0_msmarco_64way", # this model is large
        "mixedbread-ai/mxbai-colbert-large-v1",
        # "antoinelouis/colbert-xm", # XMOD based
        # "jinaai/jina-colbert-v1-en",  # requires HF token and code changes
        # BERT compatibility test only, do not recommend
        "google-bert/bert-base-uncased",
        # some colbert is trained on uncased
        # "google-bert/bert-base-cased",   # already tested uncased
    ]

    [model_embedding(model) for model in models]
