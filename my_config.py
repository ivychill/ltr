# my_config.py

import logging.handlers
import os

# def init():
# global is_training, max_eps, test_eps, logger

is_training = True         #True means Train, False means simply Run
max_eps = 1000000
test_eps = 10

LOG_PATH = './log'
MAX_LOG_SIZE = 2560000
LOG_BACKUP_NUM = 4000
logger = logging.getLogger('ltr')
log_file = os.path.join(LOG_PATH, 'ltr.log')
handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=MAX_LOG_SIZE, backupCount=LOG_BACKUP_NUM)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)