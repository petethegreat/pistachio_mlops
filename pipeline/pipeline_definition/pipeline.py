#!/usr/bin/env python 
"""pipeline.py
pipeline definition. render component definitions from templates in components directory.
define pipeline from components.
"""

from kfp import dsl
from kfp import compiler
from kfp.registry import RegistryClient

# from container_components import hyperparameter_tuning  load_data, preprocess_data, validate_data, train_monitoring
# from container_components import train_final_model, evaluate_trained_model, infer_monitoring
from components import load_data, validate_data, preprocess_data, train_monitoring, infer_monitoring, hyperparameter_tuning
from components import train_final_model, evaluate_trained_model 
from components import  evaluation_reporting, psi_result_logging

import yaml

CONFIG_FILE_PATH = '../config/config.yaml'

with open(CONFIG_FILE_PATH,'r') as config_file:
    CONFIG = yaml.safe_load(config_file)

bucket_name = CONFIG.get('gcs_bucket','the_gcs_bucket')

pipeline_root = f'gs://{bucket_name}/pistachio_pipeline_root'
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
        train_data=preprocess_train_data_task.outputs["output_file"]
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
    ).set_display_name('data monitoring - log results')

    hyperparameter_tune_task = hyperparameter_tuning(
        preprocessed_train_data=preprocess_train_data_task.outputs["output_file"],
        opt_n_iter=tuning_opt_n_iter,
        cv_seed=tuning_cv_seed
    ).set_display_name('hyperparameter tuning')

    model_train_task = train_final_model(
        preprocessed_train_data=preprocess_train_data_task.outputs["output_file"],
        optimal_parameters_json=hyperparameter_tune_task.outputs["optimal_parameters_json"],
    ).set_display_name('model training')




        # evaluate on train data task 
    evaluate_on_train_task = evaluate_trained_model(
        dataset=preprocess_train_data_task.outputs["output_file"],
        model_pickle=model_train_task.outputs['model_pickle'],
        metric_prefix="train_metrics",
        dataset_desc='Training Data'
    ).set_display_name('model evaluation - training data')
    # evaluate on test data task 

    evaluate_on_test_task = evaluate_trained_model(
        dataset=preprocess_test_data_task.outputs["output_file"],
        model_pickle=model_train_task.outputs['model_pickle'],
        metric_prefix="test_metrics",
        dataset_desc='Test Data'
    ).set_display_name('model evaluation - test data')

    # log evaluation results to vertex ai
    log_evaluation_results_task = evaluation_reporting(
        train_evaluation_results_json=evaluate_on_train_task.outputs['metric_results_json'],
        test_evaluation_results_json=evaluate_on_test_task.outputs['metric_results_json'],
        feature_importance_plot_png=evaluate_on_train_task.outputs['feature_importance_plot_png'],
        train_roc_curve_plot_png=evaluate_on_train_task.outputs['roc_curve_plot_png'],
        test_roc_curve_plot_png=evaluate_on_test_task.outputs['roc_curve_plot_png']
    ).set_display_name('log evaluation results')

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

