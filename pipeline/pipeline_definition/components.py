"""components.py
components defined using kfp container_spec
"""

from kfp import dsl
from kfp.dsl import Dataset, Input, Output, InputPath, OutputPath

import yaml 

CONFIG_FILE_PATH = '../config/default_config.yaml'

with open(CONFIG_FILE_PATH,'r') as config_file:
    CONFIG = yaml.safe_load(config_file)

BASE_IMAGE=f"{CONFIG['artifact_registry']}/{CONFIG['base_image_name']}"

@dsl.container_component
def load_data(
    input_file: str,
    output_train_path: OutputPath(Dataset),
    output_test_path: OutputPath(Dataset),
    split_seed: int=37,
    test_fraction: float=0.2,
    label_column: str='Class'
    ) -> dsl.ContainerSpec:
    """load_data component

    loads data from arff file ()

    Args:
        input_file (str): path to pistachio arff file - use /gcs fuse mount
        output_train_path (OutputPath(Dataset)): output path for train dataset (parquet)
        output_test_path (OutputPath(Dataset)): output path for test dataset (parquet)
        split_seed (int, optional): seed to be used for train/test splitting. Defaults to 37.
        test_fraction (float, optional): fraction of data to be used for test split. Defaults to 0.2.
        label_column (str, optional): column used to stratify data for splitting. Defaults to 'Class'.

    Returns:
        dsl.ContainerSpec: component definition
    """

    return dsl.ContainerSpec(
        image=BASE_IMAGE,
        command=['load_data.py'],
        args=[
            input_file,
            output_train_path,
            output_test_path,
            '--split_seed',
            split_seed,
            '--test_fraction',
            test_fraction,
            '--label_column',
            label_column
        ]
        )
#############################################################################

@dsl.container_component
def validate_data(
    input_file_path: InputPath(Dataset),
    schema_file_path: str
    ) -> dsl.ContainerSpec:
    """validate_data component

    runs pandera data validation on dataset

    Args:
        input_file_path (InputPath(Dataset)): InputPath to data to be validated
        schema_file_path (str): pandera schema file to use for validation

    Returns:
        dsl.ContainerSpec: container component definition
    """
    

    return dsl.ContainerSpec(
        image=BASE_IMAGE,
        command=['validate_data.py'],
        args=[
            input_file_path,
            schema_file_path]
        )
#############################################################################









