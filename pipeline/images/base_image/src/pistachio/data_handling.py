"""data_handling.py
functions for passing/transforming data"""

import json


import os
import logging
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np

from scipy.io import arff
from sklearn.model_selection import train_test_split
from pandera import DataFrameSchema
from pandas.api.types import CategoricalDtype

logger = logging.getLogger(__name__)

def write_to_json(content: List | Dict, path: str) -> None:
    """write_to_json
    convenience function - write content to path

    Args:
        content (List | Dict): content to be serialised
        path (str): output file location
    """

    with open(path, 'w') as outfile:
        json.dump(content, outfile)
#####################################################################

def read_from_json(path: str) -> List | Dict:
    """read_from_json
    convenience function

    Args:
        path (str): path to json file

    Returns:
        List | Dict: deserialised content
    """
    with open(path, 'r') as infile:
        content = json.load(infile)
    return content
#####################################################################

def arff_to_parquet(input_arff: str, output_parquet: str) -> None:
    """convert arff file to parquet"""
    if not os.path.exists(input_arff):
        raise ValueError(f"input file '{input_arff}' does not exist")
    logger.info(f'loading arff file {input_arff}')
    data, meta = arff.loadarff(input_arff)
    logger.info(f"arff metadata: {meta}")
    df = pd.DataFrame(data)
    df['Class'] = df['Class'].astype(str)
    df.to_parquet(output_parquet)
    logger.info(f'wrote to parquet at {output_parquet}')
##################

def parquet_to_csv(input_parquet: str, output_csv: str) -> None:
    """convert parquet to csv"""
    if not os.path.exists(input_parquet):
        raise ValueError(f"input file '{input_parquet}' does not exist")
    logger.info(f'loading parquet file {input_parquet}')
    df = pd.read_parquet(input_parquet)
    # df['Class'] = df['Class'].astype(str)
    df.to_csv(output_csv, index=False, header=True)
    logger.info(f'wrote to csv at {output_csv}')
##################

def load_arff_file(input_arff: str) -> pd.DataFrame:
    """convert arff file to parquet"""
    if not os.path.exists(input_arff):
        raise ValueError(f"input file '{input_arff}' does not exist")
    logger.info(f'loading arff file {input_arff}')
    data, meta = arff.loadarff(input_arff)
    logger.info(f"arff metadata: {meta}")
    df = pd.DataFrame(data)
    df['Class'] = df['Class'].astype(str)
    return df
##################

def load_parquet_file(input_file_path: str) -> pd.DataFrame:
    """load pandas dataframe from parquet file

    Args:
        input_file_path (str): input data file

    Returns:
        pd.DataFrame: dataframe
    """
    df = pd.read_parquet(input_file_path)
    return df
##################

def split_data(
        input_dataframe: pd.DataFrame,
        train_filename: str,
        test_filename: str,
        label_column: str,
        test_fraction: float=0.2,
        seed: int=42) -> None:
    """stratify sample the data"""
    # set seed
    # np.random.seed(seed)
    y = input_dataframe.pop(label_column)
    x_train, x_test, y_train, y_test = train_test_split(
        input_dataframe,
        y,
        random_state=seed,
        stratify=y,
        test_size=test_fraction)
    # reattach labels
    x_train[label_column] = y_train
    x_test[label_column] = y_test
    logger.info(f'x_train shape = {x_train.shape}')
    logger.info(f'y_train shape = {y_train.shape}')
    logger.info(f'x_test shape = {x_test.shape}')
    logger.info(f'y_test shape = {y_test.shape}')
    # write data
    for the_path in [train_filename, test_filename]:
        output_dir = os.path.dirname(the_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    x_train.to_parquet(train_filename)
    x_test.to_parquet(test_filename)
    logger.info(f"wrote train data to {train_filename}")
    logger.info(f"wrote test data to {test_filename}")
##############################

def validate_data_with_schema(in_df: pd.DataFrame, schema_file: str) -> pd.DataFrame:
    """check input data, count nulls, basic stats"""
    # load schema
    logger.info('validating file')
    the_schema = DataFrameSchema.from_json(schema_file)
    the_schema.validate(in_df)


def preprocess(in_raw_df: pd.DataFrame, with_target: bool=True) -> Tuple[pd.DataFrame, List[str]]:
    """preprocess the data, do any cleaning, feature engineering, etc
    handle target variable if with_target = True, else ignore/discard it
    """
    out_df = in_raw_df.copy()

    #cross some features
    out_df['SOLIDITY_MAJOR'] = out_df.SOLIDITY*out_df.MAJOR_AXIS

    # reorder
    cols = [x for x in out_df.columns if x != 'Class']

    if with_target:
        out_df = out_df[cols + ['Class']]
        # convert Class to categorical
        class_type = CategoricalDtype(categories=['Siit_Pistachio', 'Kirmizi_Pistachio'])
        out_df.Class = out_df.Class.astype(class_type)
        # create a binary column
        out_df['Target'] = out_df.Class.cat.codes
    else: 
        out_df = out_df[cols]

    return out_df, cols



