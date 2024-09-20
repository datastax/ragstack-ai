import cassio
import json
import time
import os
from glob import glob
from typing import Any, Dict, List, Generator, Tuple

from langchain_core.documents import Document
from ragstack_knowledge_store.graph_store_tags import CONTENT_ID
from langchain_core.graph_vectorstores.links import add_links, get_links, Link
from ragstack_knowledge_store.keybert_link_extractor import KeybertLinkExtractor
from ragstack_knowledge_store.langchain_cassandra_tags import CassandraGraphVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings

from keyphrase_vectorizers import KeyphraseCountVectorizer

import tiktoken
from dotenv import load_dotenv
from tqdm import tqdm
import re

from cassio.config import check_resolve_keyspace, check_resolve_session

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
BATCH_SIZE = 250
KEYSPACE = "legal_graph_store"
TABLE_NAME = "tag_based"
DRY_RUN = False

load_dotenv()

def delete_all_files_in_folder(folder_path: str) -> None:
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

token_counter = tiktoken.encoding_for_model(EMBEDDING_MODEL)
def token_count(text: str) -> int:
    return len(token_counter.encode(text))

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    return_each_line=False,
    strip_headers=False
)

# Define the regex pattern
outgoing_section_pattern = r"(\d+\.\d+)\s+\*\*(.*?)\*\*"
incoming_internal_section_pattern = r"\*\*Section\s(\d+\.\d+)\*\*"
incoming_external_section_pattern1 = r"\*\*(.*?)\s\((.*?)\),\sSection\s(\d+\.\d+)\*\*"
incoming_external_section_pattern2 = r"\*\*Section\s(\d+\.\d+)\sof\sthe\s(.*?)\s\((.*?)\)\*\*"

# Others to fix in the original dataset:
# Section 2.3
# Section 2.1
# **Section 5** of the Software Development Agreement
# **Section 4.1** of that Agreement

keybert_link_extractor = KeybertLinkExtractor(
    extract_keywords_kwargs={
        "vectorizer": KeyphraseCountVectorizer(stop_words="english"),
        "use_mmr":True,
        "diversity": 0.7
    }
)

def build_document_batch(doc_batch: List[Document]) -> List[Document]:
    keybert_links_batch = keybert_link_extractor.extract_many(doc_batch)
    for keybert_links, doc in zip(keybert_links_batch, doc_batch):
        # drop links with one word
        # pruned_links = [link for link in keybert_links if " " in link.tag]
        add_links(doc, keybert_links)
    return doc_batch


def load_chunks(markdown_file_paths: List[str]) -> Generator[List[Document], None, None]:
    doc_batch: List[Document] = []

    for markdown_file_path in tqdm(markdown_file_paths):
        with open(markdown_file_path, 'r') as file:
            markdown_text = file.read()

            docs = markdown_splitter.split_text(markdown_text)

            for doc in docs:
                doc.metadata[CONTENT_ID] = markdown_file_path

                doc_title = doc.metadata.get("Header 1", "")

                section_links = []

                # find outgoing links
                for out_section in re.findall(outgoing_section_pattern, doc.page_content):
                    out_number = out_section[0]
                    out_title = out_section[1]
                    section_links.append(Link("section", direction="in", tag=f"{doc_title} {out_number}"))

                # find incoming links
                for in_number in re.findall(incoming_internal_section_pattern, doc.page_content):
                    section_links.append(Link("section", direction="out", tag=f"{doc_title} {in_number}"))

                for in_section1 in re.findall(incoming_external_section_pattern1, doc.page_content):
                    in_title1 = in_section1[0]
                    in_abbreviation1 = in_section1[1]
                    in_number1 = in_section1[2]
                    section_links.append(Link("section", direction="out", tag=f"{in_title1} ({in_abbreviation1}) {in_number1}"))

                for in_section2 in re.findall(incoming_external_section_pattern2, doc.page_content):
                    in_number2 = in_section2[0]
                    in_title2 = in_section2[1]
                    in_abbreviation2 = in_section2[2]
                    section_links.append(Link("section", direction="out", tag=f"{in_title2} ({in_abbreviation2}) {in_number2}"))

                add_links(doc, section_links)


                doc_batch.append(doc)

                if len(doc_batch) == BATCH_SIZE:
                    yield build_document_batch(doc_batch=doc_batch)
                    doc_batch: List[Document] = []

    yield build_document_batch(doc_batch=doc_batch)

def init_graph_store() -> CassandraGraphVectorStore:
    cassio.init(auto=True)

    session = check_resolve_session()
    keyspace = check_resolve_keyspace(KEYSPACE)
    statement = session.prepare(f"DROP TABLE IF EXISTS {keyspace}.{TABLE_NAME};")
    session.execute(statement)

    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return CassandraGraphVectorStore(
        node_table=TABLE_NAME,
        session=session,
        embedding=embedding_model,
        keyspace=keyspace,
    )

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Link):
            return {"direction": obj.direction, "kind": obj.kind, "tag": obj.tag}
        return super().default(obj)

def sort_dict_by_count_and_key(data: dict[str, int]) -> List[Tuple[str, int]]:
    return sorted(data.items(), key=lambda item: (-item[1], item[0]))


def load_and_insert_chunks(dry_run: bool = True):
    in_links = set()
    out_links = set()
    bidir_links: Dict[str, int] = {}

    if dry_run:
        delete_all_files_in_folder("chunk_debug")
    else:
        graph_store = init_graph_store()

    markdown_file_paths = glob(pathname="datasets2/legal_documents/**/*.md", recursive=True)

    index = 0

    for chunk_batch in load_chunks(markdown_file_paths=markdown_file_paths):
        if not dry_run:
            while True:
                try:
                    graph_store.add_documents(chunk_batch)
                    break
                except Exception as e:
                    print(f"Encountered issue trying to store document batch: {e}")
                    time.sleep(2)
                    graph_store = init_graph_store()

        for chunk in chunk_batch:
            if dry_run:

                id = chunk.metadata[CONTENT_ID]
                id = re.sub(r'[^\w\-.]', '_', id)
                with open(f"chunk_debug/{str(index).zfill(5)}_{id}", "w") as f:
                    f.write(chunk.page_content + "\n\n")
                    f.write(json.dumps(chunk.metadata, cls=CustomJSONEncoder) + "\n\n")
                    links = get_links(chunk)
                    for link in links:
                        f.write(f"LINK Kind: '{link.kind}', Direction: '{link.direction}', Tag: '{link.tag}'\n")
                index += 1

            links = get_links(chunk)
            for link in links:
                if link.direction == "in":
                    in_links.add(link.tag)
                elif link.direction == "out":
                    out_links.add(link.tag)
                elif link.direction == "bidir":
                    if link.tag in bidir_links:
                        bidir_links[link.tag] += 1
                    else:
                        bidir_links[link.tag] = 0

    with open("debug_links_legal.json", "w") as f:
        json.dump(fp=f, obj={
            "in_links": sorted(list(in_links)),
            "out_links": sorted(list(out_links)),
            "bidir_links": sort_dict_by_count_and_key(bidir_links),
        })

    print(f"Links In: {len(in_links)}, Out: {len(out_links)}, BiDir: {len(bidir_links)}")


def main():
    load_and_insert_chunks(dry_run=DRY_RUN)
