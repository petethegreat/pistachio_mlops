# validate data
# run pandera

#! /usr/bin/env python
# load data from arff, train/test split on seed
# write train/test to parquet

from argparse import ArgumentParser
from pistachio.data_handling import load_arff_file, split_data

import logging
logger = logging

def validate_data(
    input_file_path: str,
    schema_file_location: str
    ) -> None:
    """run pandera data validation on specified file

    Args:
        input_file_path (str): _description_
        schema_file_location (str): _description_
    """
   
    df = load_arff_file(input_file_path)
    split_data(
        df,
        output_train_file_path,
        output_test_file_path,
        label_column,
        test_fraction=test_fraction,
        seed=split_seed)
    logger.info('Done splitting data')

def main():

    parser = ArgumentParser(
        description="load pistachio data from input arff file, split to train/test, write to parquet"
    )
    parser.add_argument('input_file', type='str')
    parser.add_argument('output_train_file', type='str')
    parser.add_argument('output_test_file', type='str')
    parser.add_argument('--split_seed', type='int', default=37, help='random seed for train/test split')
    parser.add_argument('--test_fraction', type='float', default=0.2, help='fraction of data to use for test set')
    parser.add_argument('--label_column', type='str', default='Class', help='label column used to stratify data')

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

if __name__ == "__main__":
    main()