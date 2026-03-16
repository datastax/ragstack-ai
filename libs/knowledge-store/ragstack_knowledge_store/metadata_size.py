import cassio
from cassio.config import check_resolve_keyspace, check_resolve_session
from dotenv import load_dotenv

load_dotenv()

KEYSPACE = "legal_graph_store"
TABLE_NAME = "metadata_based"

cassio.init(auto=True)
session = check_resolve_session()
keyspace = check_resolve_keyspace(KEYSPACE)


# Query the data from the table
rows = session.execute(f"SELECT content_id, metadata_blob FROM {keyspace}.{TABLE_NAME}")

# Loop through the rows and calculate the size of the text column
for row in rows:
    text_value = row.metadata_blob
    text_size = len(text_value.encode('utf-8'))  # Get size in bytes
    print(f"ID: {row.content_id}, Size of text_column: {text_size} bytes")

# Close the connection
session.shutdown()
