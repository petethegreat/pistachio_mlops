#! /usr/bin/env python
# load data from arff, train/test split on seed
# write train/test to parquet

from argparse import ArgumentParser
from pistachio.data_handling import load_arff_file, split_data


import logging
import sys
import os
## logging

logger = logging.getLogger('pistachio')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def load_and_split_data(
    input_file_path: str,
    output_train_file_path: str,
    output_test_file_path: str,
    split_seed: int,
    test_fraction: float,
    label_column: str) -> None:
    """load input arff file and create stratified train/test splits (parquet)

    Args:
        input_file_path (str): location of theinput file path
        output_train_file_path (str): path of output train file
        output_test_file_path (str): path of output test file
        split_seed (int): seed used for splitting data
        test_fraction (float): what fraction of data should be allocated to test
        label_column (str): label column in the input dataframe - used to stratify the split
    """

    # check input file exists
    df = load_arff_file(input_file_path)
    if not os.path.exists(input_file_path):
        logger.error(f'input arff file not found at {input_file_path}')
        sys.exit(f'input arff file not found at {input_file_path}')

    # create output directories if they do not already exist
    for path in [output_train_file_path, output_train_file_path]:
        output_dir = os.path.dirname(path)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    # split the data
    split_data(
        df,
        output_train_file_path,
        output_test_file_path,
        label_column,
        test_fraction=test_fraction,
        seed=split_seed)
    logger.info('Done splitting data')
#########################################################

def main():
    """do the things"""

    parser = ArgumentParser(
        description="load pistachio data from input arff file, split to train/test, write to parquet"
    )
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_train_file', type=str)
    parser.add_argument('output_test_file', type=str)
    parser.add_argument('--split_seed', type=int, default=37, help='random seed for train/test split')
    parser.add_argument('--test_fraction', type=float, default=0.2, help='fraction of data to use for test set')
    parser.add_argument('--label_column', type=str, default='Class', help='label column used to stratify data')

    # arff_filepath = './data/Pistachio_Dataset/Pistachio_16_Features_Dataset/Pistachio_16_Features_Dataset.arff'
    # parquet_path = './data/pistachio_16.snappy.pqt'
    args = parser.parse_args()

    load_and_split_data(
        args.input_file,
        args.output_train_file,
        args.output_test_file,
        split_seed=args.split_seed,
        test_fraction=args.test_fraction,
        label_column=args.label_column
        )
#########################################################

if __name__ == "__main__":
    main()
