from dotenv import load_dotenv
from trulens_eval import Tru
import os
import time
import logging
import sys

load_dotenv()

root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

# if os.getenv("TRULENS_DB_CONN_STRING"):
#    tru = Tru(database_url=os.getenv("TRULENS_DB_CONN_STRING"))
# else:
tru = Tru()

tru.start_evaluator()

while True:
    time.sleep(0.1)
