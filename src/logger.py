import logging
import os

def setup_basic_logger(log_file: str, level=logging.DEBUG):

    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),           
            logging.FileHandler(log_file)      
        ]
    )
    




