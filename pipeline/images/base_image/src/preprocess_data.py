#! /usr/bin/env python
# preprocess data - cleaning/feature engineering. load into format for model training/consumption

from argparse import ArgumentParser
from pistachio.data_handling import load_parquet_file, preprocess
import os

import logging
import sys
## logging

logger = logging.getLogger('pistachio')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def preprocess_data(
    input_file_path: str,
    output_file_path: str) -> None:
    """do the preprocessing

    Args:
        input_file_path (str): path to input file
        output_file_path (str): path where output will be written
    """
    logger.info(f'loading data from {input_file_path}')
    
    input_df = load_parquet_file(input_file_path)
    logger.info('preprocessing')
    output_df = preprocess(input_df)

    output_dir = os.path.dirname(output_file_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_df.to_parquet(output_file_path)
    logger.info(f'wrote processed data to {output_file_path}')
########################################################

def main():
    """do the things"""

    parser = ArgumentParser(
        description="preprocess data for model training/inference"
    )
    parser.add_argument('input_raw_file', type=str)
    parser.add_argument('output_preprocessed_file', type=str)
    

    # arff_filepath = './data/Pistachio_Dataset/Pistachio_16_Features_Dataset/Pistachio_16_Features_Dataset.arff'
    # parquet_path = './data/pistachio_16.snappy.pqt'
    args = parser.parse_args()

    preprocess_data(args.input_raw_file, args.output_preprocessed_file)

if __name__ == '__main__':
    main()
        
        
        