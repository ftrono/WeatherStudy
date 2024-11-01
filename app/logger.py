import os, logging
import logging.handlers

# Print internal logs (logger.debug)
DEBUG: bool = False

#LOG
LOG_DIR = os.path.join('logs')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

#max size of log file:
MAX_BYTES = 20*1024*1024   #20 MBytes
MAX_BACKUPS_LOGS = 5
LOG_FILE = os.path.join(LOG_DIR, 'events.log')

LOG = logging.getLogger('events')

#hdl = logging.FileHandler(LOG_FILE, mode='a')
hdl = logging.handlers.RotatingFileHandler(LOG_FILE, mode='a', maxBytes = MAX_BYTES, backupCount = MAX_BACKUPS_LOGS, encoding="utf-8", delay=False)
hdl.setFormatter(logging.Formatter('%(levelname)s: %(asctime)s f=%(funcName)s %(filename)s:%(lineno)d = %(message)s', datefmt="%Y-%m-%d %H-%M-%S"))
LOG.addHandler(hdl)

if DEBUG:
    LOG.setLevel(logging.DEBUG)
else:
    LOG.setLevel(logging.INFO)

LOG.info("Logger start...")
