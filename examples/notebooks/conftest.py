import os
import uuid

os.environ["ASTRA_DB_KEYSPACE"] = "default_keyspace"
os.environ["ASTRA_DB_COLLECTION"] = f"astra_demo_{str(uuid.uuid4()).split('-')[0]}"

# vertex-ai
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    with open("/tmp/gcloud-account-key.json", "w") as f:
        f.write(os.getenv("GCLOUD_ACCOUNT_KEY_JSON", ""))
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcloud-account-key.json"

