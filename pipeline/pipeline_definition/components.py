"""components.py
components defined using kfp container_spec
"""

from kfp import dsl
from kfp.dsl import Dataset, Input, Output, InputPath, OutputPath, Artifact, Markdown, Metrics

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
    output_file: Output[Dataset],
    feature_list: Output[Artifact]
    ) -> dsl.ContainerSpec:
    """preprocess_data component

    Args:
        input_file (Input[Dataset]): path to raw data to be preprocessed
        output_file (Output[Dataset]): path where preprocessed data will be written 
        feature_list (Output[Artifact]): path to where list of feature columns will be written as json

    Returns:
        dsl.ContainerSpec: container component definition
    """    

    return dsl.ContainerSpec(
        image=base_image_location,
        command=['./preprocess_data.py'],
        args=[
            input_file.path,
            output_file.path,
            feature_list.path]
        )
#############################################################################

@dsl.container_component
def train_monitoring(
    train_data: Input[Dataset],
    psi_artifact: Output[Artifact]
    ) -> dsl.ContainerSpec:
    """train_monitoring component

    Args:
        train_data (Input[Dataset]): preprocessed training data
        psi_artifact (Output[Artifact]): PSI artifact containing trained PSIMetrics object

    Returns:
        dsl.ContainerSpec: container component definition
    """

    return dsl.ContainerSpec(
        image=base_image_location,
        command=['./train_monitoring.py'],
        args=[
            train_data.path,
            psi_artifact.path,
            ]
        )
#############################################################################

@dsl.container_component
def infer_monitoring(
    inference_data: Input[Dataset],
    psi_artifact: Input[Artifact],
    psi_results_json: Output[Artifact]
    ) -> dsl.ContainerSpec:
    """inference monitoring component
    check for data drift when running inference

    Args:
        inference_data (Input[Dataset]): Dataset to be used for model inference
        psi_artifact (Input[Artifact]): PSI object containing statistics computed at training time
        psi_results_json (Output[Artifact]): PSI results as json file

    Returns:
        dsl.ContainerSpec: _description_
    """
   
    return dsl.ContainerSpec(
        image=base_image_location,
        command=['./infer_monitor.py'],
        args=[
            inference_data.path,
            psi_artifact.path,
            psi_results_json.path]
        )
#############################################################################

@dsl.container_component
def hyperparameter_tuning(
    preprocessed_train_data: Input[Dataset],
    featurelist_json: Input[Artifact],
    tuning_results_json: Output[Artifact],
    optimal_parameters_json: Output[Artifact],
    cv_seed: int=43,
    cv_n_folds: int=5,
    opt_n_init: int=10,
    opt_n_iter: int=200,
    opt_random_seed: int=73
    ) -> dsl.ContainerSpec:
    """hyperparameter tuning component
    tunes an CGB classifier using bayesopt to search hyperparameter space

    Args:
        preprocessed_train_data (Input[Dataset]): path to preprocessed training dataset (parquet)
        featurelist_json (Input[Artifact]): path to list of features (json)
        tuning_results_json (Output[Artifact]): output path for tuning results/details (json)
        optimal_parameters_json (Output[Artifact]): output path to best parameter set found (json)
        cv_seed (int, optional): seed used for splitting fold definition in cross validation. Defaults to 43.
        cv_n_folds (int, optional): number of folds for cross validation. Defaults to 5.
        opt_n_init (int, optional): number of initial (random) trials, prior to optimised searching. Defaults to 10.
        opt_n_iter (int, optional): number of search trials to run. Defaults to 200.
        opt_random_seed (int, optional): random seed to be used during search process. Defaults to 73.

    Returns:
        dsl.ContainerSpec: containerspec for this component
    """

    return dsl.ContainerSpec(
        image=base_image_location,
        command=['./model_tuning.py'],
        args=[
            preprocessed_train_data.path,
            featurelist_json.path,
            tuning_results_json.path,
            optimal_parameters_json.path,
            "--cv_seed", cv_seed,
            "--cv_n_folds", cv_n_folds,
            "--opt_n_init", opt_n_init,
            "--opt_n_iter", opt_n_iter,
            "--opt_random_seed", opt_random_seed]
        )
#############################################################################

@dsl.component
def psi_result_logging(
    psi_results_json: Input[Artifact],
    psi_markdown: Output[Markdown],
    psi_metrics: Output[Metrics],
    md_note: str = '',
    metric_prefix: str = 'psi_value'
    ):
    """psi_result_logging
    Generate markdown output and log metrics from json file containing psi_results

    Args:
        psi_results_json (Input[Artifact]): Json output produced when running psi evaluation
        psi_markdown (Output[Markdown]): output markdown content
        psi_metrics (Output[Metrics]): output metric artifact - psi details will be logged to this
        md_note (str): optional note/text to include in markdown
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

