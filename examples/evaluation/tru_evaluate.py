import logging
import sys
import time

import tru_shared

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

tru = tru_shared.init_tru()
tru.start_evaluator()

while True:
    time.sleep(0.1)
