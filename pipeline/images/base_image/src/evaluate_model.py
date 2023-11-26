#!/usr/bin/env python
# validate data
# run pandera

#! /usr/bin/env python
# load data from arff, train/test split on seed
# write train/test to parquet

from argparse import ArgumentParser
from pistachio.data_handling import load_parquet_file, validate_data_with_schema

import logging
import sys

## logging
logger = logging.getLogger('pistachio')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

#########################################################

def evaluate_model():
    """_summary_
    """
#########################################################
def main():

    parser = ArgumentParser(
        description="load pistachio data from input arff file, split to train/test, write to parquet"
    )
    parser.add_argument('input_file_path', type=str)
    parser.add_argument('schema_file_path', type=str)

    args = parser.parse_args()

    evaluate_model(

        )

if __name__ == "__main__":
    main()