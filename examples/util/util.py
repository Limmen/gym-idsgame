import io
import json
import jsonpickle
import os
import logging
import tqdm
import time
import argparse
from gym_idsgame.config.client_config import ClientConfig

def setup_logger(name:str, logdir:str):
    # create formatter
    formatter = logging.Formatter('%(asctime)s,%(message)s')
    # log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # log to file
    fh = logging.FileHandler(logdir + "/" + str(time.time()) + "_" + name + ".log", mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    #logger.addHandler(ch)
    return logger

def write_config_file(config, path):
    json_str = json.dumps(json.loads(jsonpickle.encode(config)), indent=4, sort_keys=True)
    with io.open(path, 'w', encoding='utf-8') as f:
        f.write(json_str)


def read_config(config_path) -> ClientConfig:
    """
    Reads configuration of the experiment from a json file

    :param config_path: the path to the configuration file
    :return: the configuration
    """
    with io.open(config_path, 'r', encoding='utf-8') as f:
        json_str = f.read()
    client_config: ClientConfig = jsonpickle.decode(json_str)
    return client_config


def parse_args(default_config_path):
    """
    Parses the commandline arguments with argparse

    :param default_config_path: default path to config file
    """
    parser = argparse.ArgumentParser(description='Parse flags to configure the json parsing')
    parser.add_argument("-cp", "--configpath", help="Path to configuration file",
                        default=default_config_path, type=str)
    args = parser.parse_args()
    return args