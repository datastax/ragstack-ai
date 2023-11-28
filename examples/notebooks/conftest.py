import os
import uuid

os.environ["ASTRA_DB_KEYSPACE"] = "ragstacke2e"
os.environ["ASTRA_DB_COLLECTION"] = f"astra_demo_{str(uuid.uuid4()).split('-')[0]}"
