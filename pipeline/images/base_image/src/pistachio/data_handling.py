
from scipy.io import arff 
import pandas as pd
import os
import logging

from typing import List
import numpy as np
from sklearn.model_selection import train_test_split

from pandera import DataFrameSchema

logger = logging.getLogger(__name__)

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

    
    
    # summary = in_df.describe(include='all')
    # # check for entirely missing columns
    # entirely_missing = [x for x in in_df.columns if summary.loc['count', x] == 0]
    # if entirely_missing:
    #     raise ValueError(f'following columns in supplied data are missing: {entirely_missing}')
    # # check that columns have more than one unique value
    # single_value_columns = [x for x in in_df.columns if summary.loc['unique', column] == 1]
    # if entirely_missing:
    #     raise ValueError(f'following columns in supplied data are missing: {entirely_missing}')


