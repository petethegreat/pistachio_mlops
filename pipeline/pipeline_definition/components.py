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
#############################################################################

@dsl.component
def psi_result_logging(
    psi_results_json: Input[Artifact],
    psi_markdown: Output[MarkDown],
    psi_metrics: Output[Metrics],
    md_note: str = '',
    metric_prefix: str = 'psi_value'
    ) -> dsl.ContainerSpec:
    """psi_result_logging
    Generate markdown output and log metrics from json file containing psi_results

    Args:
        psi_results_json_path (Input[Artifact]): _description_
        psi_markdown_path (Output[MarkDown]): _description_
        metric_prefix (str, optional): _description_. Defaults to 'psi_value'.

    Returns:
        dsl.ContainerSpec: component definition
    """
    import json
    import os
    import logging
    import sys
    logger = logging.getLogger('pistachio.psi_result_logging')
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


    # load the json content
    with open(psi_results_json.path,'r') as infile:
        psi_details = json.load(infile)
    
    # setup a string for markdown content
    # include a table header
    markdown_content = f"""
    # PSI results

    Population Stability Index evaluation

    {md_note}

    | Column | Datatype | Missing Values | PSI |"""

    # log psi metrics
    for column_name in psi_details.keys():
        the_dtype = psi_details['column_name'].get('datatype','unknown')
        n_missing = psi_details['column_name'].get('eval_missing',' ')
        psi_value = psi_details['column_name'].get('PSI','')
        table_content = f'|{column_name} | {the_dtype} | {n_missing} | {psi_value} |\n'

        # add to table
        markdown_content += table_content

        metric_name = f'{metric_prefix}_{column_name}'
        try:
            psi_metrics.log_metric(metric_name, float(psi_value))
            logger.info(f'logged {metric_name} to metrics')
        except Exception as e:
            logger.warning(f'could not log {metric_name} with value "{psi_value}" ')
        
    # write markdown content
    output_dir = os.path.dirname(psi_markdown.path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(psi_markdown.path,'w') as outfile:
        outfile.write(markdown_content)
        logger.info(f'markdown written to {psi_markdown.path}')
    logger.info('done psi result logging')
#############################################################################

