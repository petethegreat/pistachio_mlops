#!/usr/bin/env python 
"""pipeline.py
pipeline definition. render component definitions from templates in components directory.
define pipeline from components.
"""

from kfp import dsl
from kfp import compiler
from kfp.registry import RegistryClient

from components import load_data, validate_data, preprocess_data, psi_result_logging
from components import hyperparameter_tuning, train_monitoring, infer_monitoring

import yaml

CONFIG_FILE_PATH = '../config/config.yaml'

with open(CONFIG_FILE_PATH,'r') as config_file:
    CONFIG = yaml.safe_load(config_file)

bucket_name = CONFIG.get('gcs_bucket','the_gcs_bucket')

pipeline_root = f'gs://{bucket_name}/pistachio_pipeline_root/'
pipeline_name = CONFIG.get('training_pipeline_name','the_pipeline_name')
schema_file_path = f"/gcs/{bucket_name}/pipeline_resources/pistachio_schema.json"
arff_file_path = f"/gcs/{bucket_name}/pipeline_resources/Pistachio_16_Features_Dataset.arff"
stratify_column_name = 'Class'

@dsl.pipeline(
    name='pistachio_training_pipeline',
    description='pipeline for training pistachio classifier',
    pipeline_root=pipeline_root)
def pistachio_training_pipeline(
    train_test_split_seed: int=37,
    test_split_data_fraction: float=0.2,
    tuning_cv_seed: int=73,
    tuning_opt_n_iter: int=200
    ):
    """training pipeline

    Args:
        train_test_split_seed (int): _description_
        test_split_data_fraction (float): _description_
    """
    

    load_data_task = load_data(
        input_file_path=arff_file_path,
        split_seed=train_test_split_seed,
        test_fraction=test_split_data_fraction,
        label_column=stratify_column_name)
    
    validate_train_data_task = validate_data(
        input_file=load_data_task.outputs['output_train'],
        schema_file_path=schema_file_path)\
        .set_display_name('validate training data')

    validate_test_data_task = validate_data(
        input_file=load_data_task.outputs['output_test'],
        schema_file_path=schema_file_path)\
        .set_display_name('validate test data')

    preprocess_train_data_task = preprocess_data(
        input_file=load_data_task.outputs['output_train'])\
        .after(validate_train_data_task)\
        .set_display_name('preprocess train data')
    
    train_monitoring_task = train_monitoring(
        preprocess_train_data_task.outputs["output_file"]
    ).set_display_name('compute monitoring statistics')

    preprocess_test_data_task = preprocess_data(
        input_file=load_data_task.outputs['output_test'])\
        .after(validate_test_data_task)\
        .set_display_name('preprocess test data')
    
    infer_monitor_test_task = infer_monitoring(
        inference_data=preprocess_test_data_task.outputs["output_file"],
        psi_artifact=train_monitoring_task.outputs["psi_artifact"])\
        .set_display_name('test data PSI monitoring')
    
    test_monitor_logging_task = psi_result_logging(
        psi_results_json=infer_monitor_test_task.outputs['psi_results_json'],
        md_note="""logging PSI metrics at training time on test dataset"""
    )
    
    hyperparameter_tune_task = hyperparameter_tuning(
        preprocessed_train_data=preprocess_train_data_task.outputs["output_file"],
        featurelist_json=preprocess_train_data_task.outputs["feature_list"],
        opt_n_iter=tuning_opt_n_iter,
        cv_seed=tuning_cv_seed
    )
    

pipeline_output_path = './pipeline_artifact/pistaciho_training_pipeline.yaml'   
compiler.Compiler().compile(pistachio_training_pipeline, package_path=pipeline_output_path)
print(f"pipeline compiled: {pipeline_output_path}")
# upload
pipeline_registry = CONFIG.get('pipeline_registry')
if pipeline_registry:
        
    client = RegistryClient(host=pipeline_registry)

    template_name, version_name = client.upload_pipeline(
    file_name=pipeline_output_path,
    tags=["v1", "latest"],
    extra_headers={"description":"pistachio pipeline artifact"})
    print(f"uploaded pipeline to registry {pipeline_registry}")
    print(f"template_name: {template_name}")
    print(f"version_name: {version_name}")

