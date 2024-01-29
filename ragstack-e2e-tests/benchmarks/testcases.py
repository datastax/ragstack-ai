import os
import sys
import logging


from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

document = """
The term “generative AI” refers to computational techniques that are capable of generating seemingly new, meaningful content such as text, images, or audio from training
data. The widespread diffusion of this technology with examples such as Dall-E 2,
GPT-4, and Copilot is currently revolutionizing the way we work and communicate
with each other. In this article, we provide a conceptualization of generative AI as an
entity in socio-technical systems and provide examples of models, systems, and applications. Based on that, we introduce limitations of current generative AI and provide
an agenda for Business & Information Systems Engineering (BISE) research. Different
from previous works, we focus on generative AI in the context of information systems,
and, to this end, we discuss several opportunities and challenges that are unique to the
BISE community and make suggestions for impactful directions for BISE research.
Keywords: Generative AI; Artificial intelligence; Decision support; Content creation; Information systems
1 Introduction
Tom Freston is credited with saying “Innovation is taking two things that exist and putting them together
in a new way”. For a long time in history, it has been the prevailing assumption that artistic, creative
tasks such as writing poems, creating software, designing fashion, and composing songs could only
be performed by humans. This assumption has changed drastically with recent advances in artificial
1
arXiv:2309.07930v1 [cs.AI] 13 Sep 2023
intelligence (AI) that can generate new content in ways that cannot be distinguished anymore from human
craftsmanship.
The term generative AI refers to computational techniques that are capable of generating seemingly
new, meaningful content such as text, images, or audio from training data. The widespread diffusion
of this technology with examples such as Dall-E 2, GPT-4, and Copilot is currently revolutionizing the
way we work and communicate with each other. Generative AI systems can not only be used for artistic purposes to create new text mimicking writers or new images mimicking illustrators, but they can
and will assist humans as intelligent question-answering systems. Here, applications include information technology (IT) help desks where generative AI supports transitional knowledge work tasks and
mundane needs such as cooking recipes and medical advice. Industry reports suggest that generative
AI could raise global gross domestic product (GDP) by 7% and replace 300 million jobs of knowledge
workers (Goldman Sachs, 2023). Undoubtedly, this has drastic implications not only for the Business
& Information Systems Engineering (BISE) community, where we will face revolutionary opportunities,
but also challenges and risks that we need to tackle and manage to steer the technology and its use in a
responsible and sustainable direction.
In this Catchword article, we provide a conceptualization of generative AI as an entity in sociotechnical systems and provide examples of models, systems, and applications. Based on that, we introduce limitations of current generative AI and provide an agenda for BISE research. Previous papers
discuss generative AI around specific methods such as language models (e.g., Dwivedi et al., 2023; Schöbel et al., 2023; Teubner et al., 2023) or specific applications such as marketing (e.g., Peres et al., 2023),
innovation management (Burger et al., 2023), scholarly research (e.g., Davison et al., 2023; Susarla et al.,
2023), and education (e.g., Gimpel et al., 2023; Kasneci et al., 2023). Different from these works, we
focus on generative AI in the context of information systems, and, to this end, we discuss several opportunities and challenges that are unique to the BISE community and make suggestions for impactful
directions for BISE research.
"""

N_DOCS = 100


def _embedding_doc(embeddings: Embeddings, chunk_size: int):
    text_splitter = CharacterTextSplitter(
        separator="\n\n", chunk_size=chunk_size, chunk_overlap=0
    )

    split_texts = text_splitter.split_text(document)
    docs = []
    while True:
        if len(docs) == N_DOCS:
            break
        for split in split_texts:
            if len(docs) == N_DOCS:
                break
            docs.append(split)

    embeddings.embed_documents(docs)


def embeddings_batch1_chunk256(embeddings_fn):
    _embedding_doc(embeddings_fn(1), 256)


def embeddings_batch1_chunk512(embeddings_fn):
    _embedding_doc(embeddings_fn(1), 512)


def embeddings_batch10_chunk256(embeddings_fn):
    _embedding_doc(embeddings_fn(10), 256)


def embeddings_batch10_chunk512(embeddings_fn):
    _embedding_doc(embeddings_fn(10), 512)


def embeddings_batch50_chunk256(embeddings_fn):
    _embedding_doc(embeddings_fn(50), 256)


def embeddings_batch50_chunk512(embeddings_fn):
    _embedding_doc(embeddings_fn(50), 512)


def embeddings_batch100_chunk256(embeddings_fn):
    _embedding_doc(embeddings_fn(100), 256)


def embeddings_batch100_chunk512(embeddings_fn):
    _embedding_doc(embeddings_fn(100), 512)


def openai_ada002(batch_size):
    return OpenAIEmbeddings(
        chunk_size=batch_size, api_key=os.environ.get("OPEN_AI_KEY")
    )


def nvidia_nvolveqa40k(batch_size):
    # 50 is the max supported batch size
    return NVIDIAEmbeddings(
        model="nvolveqa_40k", max_batch_size=min(50, batch_size), model_type="query"
    )


if __name__ == "__main__":
    try:
        logs_file = sys.argv[1]
        logging.basicConfig(filename=logs_file, encoding="utf-8", level=logging.INFO)
        logging.info("Starting test case")
        test_case = sys.argv[2]
        embeddings = sys.argv[3]
        eval(f"{test_case}({embeddings})")
    except Exception as e:
        logging.exception("Exception in test case")
        logging.exception(e)
        raise
