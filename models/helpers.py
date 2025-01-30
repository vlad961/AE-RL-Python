import io
import logging
import os
import requests
import tensorflow as tf

from datetime import datetime
from models.agent import Agent
##################
# Helper Methods #
##################
cwd = os.getcwd()

def download_file(url:str, local_filename:str):
    """
    Download a file from a given URL and save it locally.

    Args:
        url (str): The URL of the file to download.
        local_filename (str): The local path where the file will be saved.

    Returns:
        str: The local path where the file was saved.
    """
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def save_model(agent: Agent, model_path):
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    # Rename the model
    stream = io.StringIO()
    agent.model_network.model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_str = stream.getvalue()
    stream.close()
    agent.model_network.model.save(model_path)
    logging.info("Model summary:\n{}".format(summary_str))
    logging.info("Model saved in: {}".format(model_path))


def load_model(agent: Agent, model_path):
    agent.model_network.model.load_model(model_path)

def download_datasets_if_missing(kdd_train:str, kdd_test:str):
    # If the data files for some reason do not exist, download them from the repo this work is based on.
    if (not os.path.exists(kdd_train)):
        kdd_train_url = "https://raw.githubusercontent.com/gcamfer/Anomaly-ReactionRL/master/datasets/NSL/KDDTrain%2B.txt"
        download_file(kdd_train_url, kdd_train)
        logging.info("Downloaded: {}\nSaved in: {}", kdd_train_url, kdd_train)
    if (not os.path.exists(kdd_test)):
        kdd_test_url = "https://raw.githubusercontent.com/gcamfer/Anomaly-ReactionRL/master/datasets/NSL/KDDTest%2B.txt"
        download_file(kdd_test_url, kdd_test)
        logging.info("Downloaded: {}\nSaved in: {}", kdd_test_url, kdd_test)

def logger_setup():
    timestamp_begin = datetime.now().strftime("%Y-%m-%d-%H-%M")
    # Configure logging
    log_filename = os.path.join(cwd, 'logs/{}.log'.format(timestamp_begin))
    logging.basicConfig(filename=os.path.join(cwd, log_filename), level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create a console handler for the logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Add the console handler to the logger
    logging.getLogger().addHandler(console_handler)

    # Redirect TensorFlow logs to the logging module
    tf.get_logger().setLevel('INFO')
    tf.get_logger().addHandler(console_handler)