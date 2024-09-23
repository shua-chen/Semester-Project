import logging
import os
import sys

def setup_logger(log_file='./logs/data_pre.log',name='data_pre', level=logging.INFO):
    """Function setup as many loggers as you want"""
    if os.path.exists(log_file):
        os.remove(log_file)
    
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger