from dotenv import load_dotenv
from trulens_eval import Tru
import os

load_dotenv()

tru = Tru(database_url=os.getenv("TRULENS_DB_CONN_STRING"))
tru.reset_database()
