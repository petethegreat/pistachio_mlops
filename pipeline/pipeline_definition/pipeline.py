#!/usr/bin/env python 
"""pipeline.py
pipeline definition. render component definitions from templates in components directory.
define pipeline from components.
"""

from kfp import dsl
from kfp import compiler
from components import load_data, validate_data

import yaml

CONFIG_FILE_PATH = '../config/config.yaml'

with open(CONFIG_FILE_PATH,'r') as config_file:
    CONFIG = yaml.safe_load(config_file)

bucket_name = CONFIG.get('gcs_bucket','the_gcs_bucket')

pipeline_root = f'gs://{bucket_name}/pistachio_pipeline_root/'
pipeline_name = CONFIG.get('training_pipeline_name','the_pipeline_name')


@dsl.pipeline(
    name='pistachio_training_pipeline',
    description='pipeline for training pistachio classifier',
    pipeline_root=pipeline_root)
def pistachio_training_pipeline(
    train_test_split_seed: int,
    test_split_data_fraction: float
    ):
    """training pipeline

    Args:
        train_test_split_seed (int): _description_
        test_split_data_fraction (float): _description_
    """
    
    arff_file_location = 'arff_file gcs_url or /gcs path'
    stratify_column_name = 'Class'
    schema_file_path = 'schema file path in gcs'

    load_data_task = load_data(
        input_file_path=arff_file_location,
        split_seed=train_test_split_seed,
        test_fraction=test_split_data_fraction,
        label_column=stratify_column_name)
    
    validate_train_data_task = validate_data(
        input_file=load_data_task.outputs['output_train'],
        schema_file_path=schema_file_path
        )
    validate_test_data_task = validate_data(
        input_file=load_data_task.outputs['output_test'],
        schema_file_path=schema_file_path
        )
    
compiler.Compiler().compile(pistachio_training_pipeline, package_path='./pipeline_artifact/pistaciho_training_pipeline.yaml')
