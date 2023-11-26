#!/usr/bin/env python

"""train_model.py
component for training a model and saving model artifact(s)
"""


from argparse import ArgumentParser
from pistachio.data_handling import load_parquet_file, read_from_json
from pistachio.model_training import stuff


import logging
import sys

## logging
logger = logging.getLogger('pistachio')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def setup_logging():
    """log to stdout"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def train_model(
    training_data_path: str,
    optimal_parameters_json_path: str,
    output_model_artifact_path: str,
    features_path: str
    ):
    """train an XGBClassifier

    Args:
        training_data_path (str): path to training data parquet
        optimal_parameters_json_path (str): path to json file containing optimal classifier parameters
        output_model_artifact_path (str): output location for model artifact
    """

    data = load_parquet_file(training_data_path)
    logger.info(f'read data from {training_data_path} ')
    features = read_from_json(features_path)
    parameters = read_from_json(optimal_parameters_json_path)







#########################################################

def main():
    setup_logging()

    parser = ArgumentParser(
        description="load training data and optimal parameters, train an xgboost classifier and save artifacts"
    )
    parser.add_argument('train_data', type=str)
    parser.add_argument('optimal_parameters_json', type=str)
    parser.add_argument('model_artifact', type=str)

    args = parser.parse_args()

    train_model(
        args.train_data,
        args.optimal_parameters_json,
        args.model_artifact
        )

if __name__ == "__main__":
    main()