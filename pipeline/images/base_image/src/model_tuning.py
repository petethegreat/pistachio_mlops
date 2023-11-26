#!/usr/bin/env python
"""model_tuning.py
code for model tuning component
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






def main():
    """do the things"""

    parser = ArgumentParser(
        description="search hyperparameter space to determine optimal hyperparameters for model"
    )
    parser.add_argument('input_raw_file', type=str)
    parser.add_argument('output_preprocessed_file', type=str)


    # arff_filepath = './data/Pistachio_Dataset/Pistachio_16_Features_Dataset/Pistachio_16_Features_Dataset.arff'
    # parquet_path = './data/pistachio_16.snappy.pqt'
    args = parser.parse_args()

    preprocess_data(args.input_raw_file, args.output_preprocessed_file)
if __name__ == "__main__":
    main()
