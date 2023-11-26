#! /usr/bin/env python
# load data from arff, train/test split on seed
# write train/test to parquet

from argparse import ArgumentParser
from pistachio.data_handling import load_arff_file, split_data

import logging
logger = logging

def load_and_split_data():
    """load and split data"""

    df = load_arff_file()
    split_data()
    logger.info('Done')

def main():

    parser = ArgumentParser(
        description="load pistachio data from input arff file, split to train/test, write to parquet"
    )
    parser.add_argument('input_file', type='str')
    parser.add_argument('output_train_file', type='str')
    parser.add_argument('--split_seed', type='int', default=37, help='random seed for train/test split')
    parser.add_argument('--test_fraction', type='float', default=0.2, help='fraction of data to use for test set')
    parser.add_argument('--label_column', type='str', default='Class', help='label column used to stratify data')

    arff_filepath = './data/Pistachio_Dataset/Pistachio_16_Features_Dataset/Pistachio_16_Features_Dataset.arff'
    parquet_path = './data/pistachio_16.snappy.pqt'

    load_and_split_data()


if __name__ == "__main__":
    main()