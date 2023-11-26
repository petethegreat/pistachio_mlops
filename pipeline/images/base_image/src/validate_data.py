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

def setup_logging():
    """log to stdout"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
#########################################################

def validate_data(
    input_file_path: str,
    schema_file_path: str
    ) -> None:
    """run pandera data validation on specified file

    Args:
        input_file_path (str): _description_
        schema_file_path (str): _description_
    """

  
    df = load_parquet_file(input_file_path)
    logger.info(f'loaded data from {input_file_path}')
    validate_data_with_schema(df, schema_file_path)
    logger.info('file passed validation')
#########################################################3

def main():
    setup_logging()

    parser = ArgumentParser(
        description="load pistachio data from input arff file, split to train/test, write to parquet"
    )
    parser.add_argument('input_file_path', type=str)
    parser.add_argument('schema_file_path', type=str)

    args = parser.parse_args()

    validate_data(
        args.input_file_path,
        args.schema_file_path,
        )

if __name__ == "__main__":
    main()