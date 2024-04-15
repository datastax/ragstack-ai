import logging

import pytest

from ragstack_colbert import (
    CassandraColbertVectorStore,
    ColbertCassandraRetriever,
    ColbertTokenEmbeddings,
)
from ragstack_llamaindex.colbert import ColbertVectorStoreLlamaIndexRetriever
from tests.integration_tests.conftest import (
    get_astradb_test_store,
    get_local_cassandra_test_store,
)


@pytest.fixture
def cassandra():
    return get_local_cassandra_test_store()


@pytest.fixture
def astra_db():
    return get_astradb_test_store()


@pytest.mark.parametrize("vector_store", ["cassandra", "astra_db"])
def test_embedding_cassandra_retriever(request, vector_store: str):
    vector_store = request.getfixturevalue(vector_store)
    narrative = """
    Marine animals inhabit some of the most diverse environments on our planet. From the shallow coral reefs teeming with colorful fish to the dark depths of the ocean where mysterious creatures lurk, the marine world is full of wonder and mystery.

    One of the most iconic marine animals is the dolphin, known for its intelligence, social behavior, and playful antics. Dolphins communicate with each other using a variety of clicks, whistles, and body movements. They live in social groups called pods and often display behaviors that suggest a high level of social complexity, including cooperation for hunting and care for the injured or sick members of their pod.

    Another remarkable creature is the sea turtle, which navigates vast oceans to return to the very beaches where they were born to lay their eggs. These ancient mariners are true survivors, having roamed the oceans for millions of years. However, they face numerous threats from human activities, including plastic pollution, accidental capture in fishing gear, and the loss of nesting beaches due to climate change.

    Deep in the ocean's abyss, where sunlight fails to penetrate, live the bioluminescent creatures, such as the anglerfish. These eerie-looking fish use a natural light produced by bacteria in their lure to attract prey in the pitch-black waters. This fascinating adaptation is a perfect example of the unique strategies marine animals have evolved to survive in the ocean's different layers.

    Coral reefs, often referred to as the "rainforests of the sea," are another critical habitat. They are bustling with life and serve as a vital ecosystem for many marine species. Corals themselves are fascinating organisms. They are made up of thousands of tiny creatures called polyps and have a symbiotic relationship with algae, which provides them with food through photosynthesis.

    The diversity of marine life is vast, and each species plays a crucial role in the ocean's ecosystem. From the microscopic plankton that form the base of the oceanic food chain to the majestic blue whale, the largest animal to have ever lived on Earth, marine animals are an integral part of our world's biodiversity. Protecting these creatures and their habitats is essential for maintaining the health of our oceans and the planet as a whole.
    """

    # Define the desired chunk size and overlap size
    chunk_size = 450
    overlap_size = 50

    # Function to generate chunks with the specified size and overlap
    def generate_chunks(text, chunk_size, overlap_size):
        chunks = []
        start = 0
        end = chunk_size
        while start < len(text):
            # If this is not the first chunk, move back 'overlap_size' characters to create the overlap
            if start != 0:
                start -= overlap_size
            chunks.append(text[start:end])
            start = end
            end += chunk_size
        return chunks

    # Generate the chunks based on the narrative
    chunks = generate_chunks(narrative, chunk_size, overlap_size)

    # Output the first few chunks to ensure they meet the specifications
    for i, chunk in enumerate(chunks[:3]):  # Displaying the first 3 chunks for brevity
        logging.info(f"Chunk {i + 1}:\n{chunk}\n{'-' * 50}\n")

    doc_id = "Marine Animals habitat"

    # colbert stuff starts
    colbert = ColbertTokenEmbeddings(
        doc_maxlen=220,
        nbits=1,
        kmeans_niters=4,
    )

    embedded_chunks = colbert.embed_chunks(texts=chunks, doc_id=doc_id)

    logging.info(f"embedded chunks size {len(embedded_chunks)}")

    store = CassandraColbertVectorStore(
        keyspace="default_keyspace",
        table_name="colbert_embeddings",
        session=vector_store.create_cassandra_session(),
    )
    store.put_chunks(chunks=embedded_chunks, delete_existing=True)

    retriever = ColbertCassandraRetriever(
        vector_store=store, colbert_embeddings=colbert
    )
    lc_retriever = ColbertVectorStoreLlamaIndexRetriever(retriever, k=2)
    docs = lc_retriever.get_relevant_documents(
        "what kind fish lives shallow coral reefs atlantic, india ocean, red sea, gulf of mexico, pacific, and arctic ocean"
    )
    assert len(docs) == 2
    assert len(docs[0].page_content) > 0