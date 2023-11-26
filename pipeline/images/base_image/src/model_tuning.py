#!/usr/bin/env python
"""model_tuning.py
code for model tuning component
hyperparameter search for xgboost classifier
"""


from argparse import ArgumentParser
from pistachio.data_handling import load_parquet_file
from pistachio.model_training import optimise_tune

import os

import logging
import sys
## logging

logger = logging.getLogger('pistachio')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def model_tune():
    # load preprocessed train data 
    # break into X and Y 
    # define search space - this should come from pipeline somehow, maybe as json?
    # can have argparse do it?
    """run model tuning trials"""

# pbounds = {
#     'learning_rate': (0.01, 0.3),
#     'gamma': (0.0, 0.3),
#     'min_child_weight': (0.01, 0.07),
#     'max_depth': (3, 5),
#     'subsample': (0.7, 0.9),
#     'reg_alpha': (0.01, 0.1),
#     'reg_lambda': (0.01, 0.1)
# }
def main():
    """do the things"""

    parser = ArgumentParser(
        description="search hyperparameter space to determine optimal hyperparameters for model"
    )
    parser.add_argument('input_raw_file', type=str)
    parser.add_argument('output_preprocessed_file', type=str)
    parser.add_argument('learning_rate', nargs=2, type=float, metavars=('learning_rate_low', 'learning_rate_high'), default=(0.01, 0.3))
    parser.add_argument('gamma', nargs=2, type=float, metavars=('gamma_low', 'gamma_high'), default=(0.0, 0.3))
    parser.add_argument('min_child_weight', nargs=2, type=float, metavars=('min_child_weight_low', 'min_child_weight_high'), default=(0.01, 0.07))
    parser.add_argument('max_depth', nargs=2, type=int, metavars=('max_depth_low', 'max_depth_high'), default=(3, 5))
    parser.add_argument('subsample', nargs=2, type=float, metavars=('subsample_low', 'subsample_high'), default=(0.7, 0.9))
    parser.add_argument('reg_alpha', nargs=2, type=float, metavars=('reg_alpha_low', 'reg_alpha_high'), default=(0.01, 0.1))
    parser.add_argument('reg_lambda', nargs=2, type=float, metavars=('reg_lambda_low', 'reg_lambda_high'), default=(0.01, 0.1))


    # arff_filepath = './data/Pistachio_Dataset/Pistachio_16_Features_Dataset/Pistachio_16_Features_Dataset.arff'
    # parquet_path = './data/pistachio_16.snappy.pqt'
    args = parser.parse_args()

    preprocess_data(args.input_raw_file, args.output_preprocessed_file)
if __name__ == "__main__":
    main()
