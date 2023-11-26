#!/usr/bin/env python

"""train_model.py
component for training a model and saving model artifact(s)
"""

import logging
import sys
import os

from argparse import ArgumentParser

from xgboost import XGBClassifier

from pistachio.data_handling import load_parquet_file, read_from_json
from pistachio.model_training import train_model, save_model_to_pickle



## logging
logger = logging.getLogger('pistachio')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def train_final_model(
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

    train_data = load_parquet_file(training_data_path)
    logger.info(f'read data from {training_data_path} ')
    features = read_from_json(features_path)
    parameters = read_from_json(optimal_parameters_json_path)


    train_x = train_data[features]
    train_y = train_data.Target.values

    model = train_model(train_x, train_y, parameters )

    # create output directory, if it does not already exist
    output_dir = os.path.dirname(output_model_artifact_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_model_to_pickle(model, output_model_artifact_path)

    
#########################################################

def main():

    parser = ArgumentParser(
        description="load training data and optimal parameters, train an xgboost classifier and save artifacts"
    )
    parser.add_argument('train_data', type=str)
    parser.add_argument('optimal_parameters_json', type=str)
    parser.add_argument('model_artifact', type=str)
    parser.add_argument('feature_list_json', type=str)


    args = parser.parse_args()

    train_final_model(
        args.train_data,
        args.optimal_parameters_json,
        args.model_artifact,
        args.feature_list_json
        )

if __name__ == "__main__":
    main()