#! /usr/bin/env python
# preprocess data - cleaning/feature engineering. load into format for model training/consumption

import os
import json
import logging
import sys

from argparse import ArgumentParser
from pistachio.data_handling import load_parquet_file, preprocess
from pistachio.utils import ensure_directory_exists
from typing import List
## logging

logger = logging.getLogger('pistachio')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def preprocess_data_features(
    input_file_path: str,
    output_file_path: str
    ) -> List[str]:
    """do the preprocessing

    Args:
        input_file_path (str): path to input file
        output_file_path (str): path where output will be written
        feature_list_path (str): path where list of features will be written (as json)
    """
    logger.info(f'loading data from {input_file_path}')

    input_df = load_parquet_file(input_file_path)
    logger.info('preprocessing')
    output_df, features = preprocess(input_df)

    for path in [output_file_path, feature_list_path]:
        ensure_directory_exists(path)

    output_df.to_parquet(output_file_path)
    logger.info(f'wrote processed data to {output_file_path}')
    return features
    

########################################################

def preprocess_data(
    input_file_path: str,
    output_file_path: str,
    feature_list_path: str 
    ) -> None:
    """do the preprocessing

    Args:
        input_file_path (str): path to input file
        output_file_path (str): path where output will be written
        feature_list_path (str): path where list of features will be written (as json)
    """
    logger.info(f'loading data from {input_file_path}')

    input_df = load_parquet_file(input_file_path)
    logger.info('preprocessing')
    output_df, features = preprocess(input_df)

    for path in [output_file_path, feature_list_path]:
        ensure_directory_exists(path)

    output_df.to_parquet(output_file_path)
    logger.info(f'wrote processed data to {output_file_path}')
    
    with open(feature_list_path, 'w') as outfile:
        json.dump(features, outfile)
    logger.info(f'wrote feature list to {feature_list_path}')
########################################################

def main():
    """do the things"""

    parser = ArgumentParser(
        description="preprocess data for model training/inference"
    )
    parser.add_argument('input_raw_file', type=str)
    parser.add_argument('output_preprocessed_file', type=str)
    parser.add_argument('feature_list_path', type=str)

    # arff_filepath = './data/Pistachio_Dataset/Pistachio_16_Features_Dataset/Pistachio_16_Features_Dataset.arff'
    # parquet_path = './data/pistachio_16.snappy.pqt'
    args = parser.parse_args()

    preprocess_data(args.input_raw_file, args.output_preprocessed_file, args.feature_list_path)

if __name__ == '__main__':
    main()


