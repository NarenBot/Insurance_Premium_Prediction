import os
import sys
from datetime import datetime
import logging

FORMAT = "[%(asctime)s] %(levelname)s | %(filename)s | %(lineno)d | %(message)s"
log_file_name = f"{datetime.now().strftime('%d-%b-%Y_%H-%M-%S')}.log"
log_file_path = os.path.join(os.getcwd(), "logs", log_file_name)

os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(filename=log_file_path, format=FORMAT, level=logging.INFO)

# log = logging.getLogger()
