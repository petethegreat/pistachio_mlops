"""components.py
components defined using kfp container_spec
"""

from kfp import dsl
from kfp.dsl import Dataset, Input, Output, InputPath, OutputPath

import yaml 

CONFIG_FILE_PATH = '../config/config.yaml'

with open(CONFIG_FILE_PATH,'r') as config_file:
    CONFIG = yaml.safe_load(config_file)

artifact_registry = CONFIG.get('artifact_registry','the_artifact_registry')
base_image_name = CONFIG.get('base_image_name','the_base_image:0.0.0')
base_image_location = f'{artifact_registry}/{base_image_name}'

@dsl.container_component
def load_data(
    input_file_path: str,
    output_train: Output[Dataset],
    output_test: Output[Dataset],
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
        image=base_image_location,
        command=['load_data.py'],
        args=[
            input_file_path,
            output_train.path,
            output_test.path,
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
    input_file: Input[Dataset],
    schema_file_path: str
    ) -> dsl.ContainerSpec:
    """validate_data component

    runs pandera data validation on dataset

    Args:
        input_file (InputPath(Dataset)): InputPath to data to be validated
        schema_file_path (str): pandera schema file to use for validation

    Returns:
        dsl.ContainerSpec: container component definition
    """
    

    return dsl.ContainerSpec(
        image=base_image_location,
        command=['validate_data.py'],
        args=[
            input_file.path,
            schema_file_path]
        )
#############################################################################









