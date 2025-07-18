import logging
from logging import info, debug, error
import os

LOG_PATH, LOG_FILENAME = '.', 'philharmonic.log'
LOG_LEVEL1 = logging.DEBUG
LOG_LEVEL2 = logging.INFO

# set up logging to file
logging.basicConfig(
     filename=LOG_FILENAME,
     level=LOG_LEVEL1,
     format='%(message)s'
 )

# set up logging to console
console = logging.StreamHandler()
console.setLevel(LOG_LEVEL2)
# set a format which is simpler for console use
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

logger = logging.getLogger(__name__)

def log(message):
    logging.info(message)
