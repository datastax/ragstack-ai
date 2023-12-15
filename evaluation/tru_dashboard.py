from trulens_eval import Tru
from dotenv import load_dotenv
import os

load_dotenv()

tru = Tru(database_url=os.getenv("TRULENS_DB_CONN_STRING"))
tru.run_dashboard(address="0.0.0.0", port=8501, force=True)
