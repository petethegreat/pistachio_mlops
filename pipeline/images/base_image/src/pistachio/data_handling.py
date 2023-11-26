
from scipy.io import arff 
import pandas as pd
import os
import logging

from typing import List
import numpy as np
from sklearn.model_selection import train_test_split

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

def load_arff_file(input_arff: str, output_parquet: str) -> pd.DataFrame:
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


def split_data(
        input_parquet: str, 
        train_filename: str,
        test_filename: str,
        label_column: str,
        test_fraction: float=0.2,
        seed: int=42) -> None:
    """stratify sample the data"""
    # set seed
    # np.random.seed(seed)
    in_df = pd.read_parquet(input_parquet)
    y = in_df.pop(label_column)
    x_train, x_test, y_train, y_test = train_test_split(
        in_df, 
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
##############################

