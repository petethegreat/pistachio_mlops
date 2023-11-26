"""components.py
components defined using kfp container_spec
"""

from kfp import dsl
from kfp.dsl import Dataset, Input, Output, InputPath, OutputPath, Artifact

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
        command=['./load_data.py'],
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

    Args:
        input_file (Input[Dataset]): path to input dataset to be validated
        schema_file_path (str): pandera schema file to use for validation

    Returns:
        dsl.ContainerSpec: container component definition
    """

    return dsl.ContainerSpec(
        image=base_image_location,
        command=['./validate_data.py'],
        args=[
            input_file.path,
            schema_file_path]
        )
#############################################################################

@dsl.container_component
def preprocess_data(
    input_file: Input[Dataset],
    output_file: Output[Dataset]
    ) -> dsl.ContainerSpec:
    """preprocess_data component

    preprocesses data (feature engineering, data transformation)

    Args:
        input_file (Input(Dataset)): path to raw data to be preprocessed
        output_file (Output(Dataset)): path where preprocessed data will be written 
    Returns:
        dsl.ContainerSpec: container component definition
    """
    

    return dsl.ContainerSpec(
        image=base_image_location,
        command=['./preprocess_data.py'],
        args=[
            input_file.path,
            output_file.path]
        )

@dsl.component
def psi_json_to_artifacts(
    psi_json_path: Input[Artifact],
    psi_metrics: Output[Metrics],
    psi_markdown: Output[Markdown]
    ):
    """load json file of psi result, write as markdown and log metrics


    Args:
        psi_json_path (Input[Artifact]): json file with psi details
        psi_metrics (Output[Metrics]): psi values per column as metrics
        psi_markdown (Output[Markdown]): psi values per column as markdown content.
    """
    import json
    with open(psi_json_path.path) as input:
        psi_details = json.loads(input)

    





