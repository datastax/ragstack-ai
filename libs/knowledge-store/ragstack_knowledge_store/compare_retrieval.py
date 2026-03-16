import pickle
from langchain_core.documents import Document

from typing import Dict, List

def get_stuff(table_name):
    with open(f"debug_retrieval_{table_name}.pkl", "rb") as file:
        return pickle.load(file)


metadata_based: Dict[str, List[Document]] = get_stuff("metadata_based")
link_based: Dict[str, List[Document]] = get_stuff("link_column_based")

count = 1
for query in metadata_based.keys():
    metadata_chunks = metadata_based[query]
    link_chunks = link_based[query]

    print(f"Query {count} has {len(metadata_chunks)} metadata chunks and {len(link_chunks)} link chunks. Diff: {len(metadata_chunks)-len(link_chunks)}")
    count += 1
