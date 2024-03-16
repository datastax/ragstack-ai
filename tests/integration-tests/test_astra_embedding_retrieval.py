# test 
from ragstack.colbert.colbert_embedding import ColbertTokenEmbeddings
from ragstack.colbert.astra_retriever import ColbertAstraRetriever
from ragstack.colbert.cassandra_db import AstraDB
from .cassandra_container import CassandraContainer
import os
import base64

def test_embedding_astra_retriever():

    docker_container = CassandraContainer()
    docker_container.start()
   
    # Initial narrative about marine animals to then break down into chunks as specified by the user
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
        print(f"Chunk {i+1}:\n{chunk}\n{'-'*50}\n")

    title = "Marine Animals habitat"

    # colbert stuff starts
    colbert = ColbertTokenEmbeddings(
        doc_maxlen=220,
        nbits=1,
        kmeans_niters=4,
    )

    passageEmbeddings = colbert.embed_documents(texts=chunks, title=title)

    print(f"passage embeddings size {len(passageEmbeddings)}")

    # Fetch the Base64 encoded string from the environment variable
    encoded_zip = os.getenv('COLBERT_ASTRA_SCB')

    # Decode the Base64 string back to binary data
    decoded_zip = base64.b64decode(encoded_zip)

    # Specify the output path for the rebuilt ZIP file
    output_zip_path = '/tmp/secure-connect-mingv1.zip'

    # Write the binary data to a new file
    with open(output_zip_path, 'wb') as zip_file:
        zip_file.write(decoded_zip)


    # astra db
    astra = AstraDB(
        secure_connect_bundle=output_zip_path,
        astra_token=os.getenv("COLBERT_ASTRA_TOKEN"),
        keyspace="colberttest",
        verbose=True,
    )

    astra.ping()

    print("astra db is connected")

    # astra insert colbert embeddings
    astra.insert_colbert_embeddings_chunks(
        embeddings=passageEmbeddings, delete_existed_passage=True)


    retriever = ColbertAstraRetriever(astraDB=astra, colbertEmbeddings=colbert, verbose=True)
    answers = retriever.retrieve("what kind fish lives shallow coral reefs", k=5)
    for a in answers:
        print(f"answer rank {a['rank']} score {a['score']}, answer is {a['body']}\n")
    assert len(answers) == 5

    astra.close()



