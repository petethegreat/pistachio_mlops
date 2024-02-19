#!/usr/bin/env python 
"""training_pipeline.py
pipeline definition. render component definitions from templates in components directory.
define pipeline from components.
"""

from kfp import dsl
from kfp import compiler
from kfp.registry import RegistryClient

from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.model import ModelUploadOp

# from container_components import hyperparameter_tuning  load_data, preprocess_data, validate_data, train_monitoring
# from container_components import train_final_model, evaluate_trained_model, infer_monitoring
from components import load_data, validate_data, preprocess_data, train_monitoring, infer_monitoring, hyperparameter_tuning
from components import train_final_model, evaluate_trained_model, upload_model_to_registry
from components import  evaluation_reporting, psi_result_logging, copy_artifacts_to_storage

import yaml

CONFIG_FILE_PATH = '../config/config.yaml'

with open(CONFIG_FILE_PATH,'r') as config_file:
    CONFIG = yaml.safe_load(config_file)

bucket_name = CONFIG.get('gcs_bucket','the_gcs_bucket')

pipeline_root = f'gs://{bucket_name}/pistachio_pipeline_root'
pipeline_name = CONFIG.get('training_pipeline_name','the_pipeline_name')
schema_file_path = f"/gcs/{bucket_name}/pipeline_resources/pistachio_schema.json"
arff_file_path = f"/gcs/{bucket_name}/pipeline_resources/Pistachio_16_Features_Dataset.arff"
project_id = CONFIG.get('project_id', 'the_project_id')
stratify_column_name = 'Class'

# name to use in registry
model_name = CONFIG.get('model_registry_name','pistachio_classifier')
model_registry_location = CONFIG.get('model_registry_location','the_gcp_location')
# need this for the model upload op
artifact_registry = CONFIG.get('artifact_registry', 'the_artifact_registry')
base_image_name = CONFIG.get('base_image_name', 'the_base_image:0.0.0')
base_image_location = f'{artifact_registry}/{base_image_name}'
serving_image_name = CONFIG.get('serving_image_name', 'the_serving_image:0.0.0')
serving_image_location = f'{artifact_registry}/{serving_image_name}'
artifact_path = CONFIG.get('artifact_path','the_artifact_path')


@dsl.pipeline(
    name=pipeline_name,
    description='pipeline for training pistachio classifier',
    pipeline_root=pipeline_root)
def pistachio_training_pipeline(
    train_test_split_seed: int=37,
    test_split_data_fraction: float=0.2,
    tuning_cv_seed: int=73,
    tuning_opt_n_iter: int=200,
    project_id: str=project_id
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

    # This is a headache.
    # train task outputs artifact[Model]
    # modelUploadOp wants an input of type Input[UnmanagedContainerModel]
    # can't automatically cast.
    # can import/export the model, but that needs uris/paths to be defined ahead of time (not just Outputs)
    # Could get around it by using a pipeline as a component - have a mini pipeline that takes uri as string and imports and uploads, then feed the Output[Model].uri to that
    
    # easier to just write a component using aiplatform package/functions

    # import model artifact
    # load the model artifact as an artifact of type google vertex model
    # mainly, this defines containerspec, and predict/health roots for google vertex model

    # model_artifact_import_task = dsl.importer(
    #     artifact_uri=model_train_task.outputs['model_pickle'],
    #     artifact_class=artifact_types.UnmanagedContainerModel,
    #     metadata={
    #         'containerSpec': { 
    #             'imageUri': base_image_location,
    #             'command': ['./serve_predictions.py'],
    #             # 'args': []
    #             # 'env': []
    #             # 'ports': [],
    #             "predictRoute": "/predict",
    #             "healthRoute": "/health"
    #         }
    #     },
    #     reimport=False)\
    #     .set_display_name('cast_model_artifact')\
    #     .after(model_train_task)

    # # try this
    # model_upload_task = ModelUploadOp(
    #     project=project_id,
    #     display_name='upload_model_to_registry',
    #     description='model for pistachio classification',
    #     # model_id='pistachio_classifier_model',
    #     labels={'model_name': 'pistachio_classifier_model'},
    #     # parent_model=PARENT_MODEL_ID
    #     unmanaged_container_model=model_train_task.outputs['model_pickle']
    # )

    # copy artifacts to specific location in bucket
    # upload model using artifact id from that component
    #   dummy('{{workflow.labels.pipeline/runid}}', '{{workflow.annotations.pipelines.kubeflow.org/run_name}}')

    # https://stackoverflow.com/a/71384129
    # print_op(msg='job name:', value=dsl.PIPELINE_JOB_NAME_PLACEHOLDER)
    # print_op(msg='job resource name:', value=dsl.PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER)
    # print_op(msg='job id:', value=dsl.PIPELINE_JOB_ID_PLACEHOLDER)
    # print_op(msg='task name:', value=dsl.PIPELINE_TASK_NAME_PLACEHOLDER)
    # print_op(msg='task id:', value=dsl.PIPELINE_TASK_ID_PLACEHOLDER)

    # https://github.com/kubeflow/pipelines/blob/master/sdk/python/kfp/dsl/__init__.py
    # 'PIPELINE_JOB_NAME_PLACEHOLDER',
    # 'PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER',
    # 'PIPELINE_JOB_ID_PLACEHOLDER',
    # 'PIPELINE_TASK_NAME_PLACEHOLDER',
    # 'PIPELINE_TASK_ID_PLACEHOLDER',
    # 'PIPELINE_TASK_EXECUTOR_OUTPUT_PATH_PLACEHOLDER',
    # 'PIPELINE_TASK_EXECUTOR_INPUT_PLACEHOLDER',
    # 'PIPELINE_ROOT_PLACEHOLDER',
    # 'PIPELINE_JOB_CREATE_TIME_UTC_PLACEHOLDER',
    # 'PIPELINE_JOB_SCHEDULE_TIME_UTC_PLACEHOLDER',



    copy_artifacts_to_storage_task = copy_artifacts_to_storage(
        project_id=project_id,
        storage_bucket=bucket_name,
        artifact_path=artifact_path,
        pipeline_name=pipeline_name,
        run_name='{{workflow.annotations.pipelines.kubeflow.org/run_name}}',
        run_id = dsl.PIPELINE_JOB_ID_PLACEHOLDER,
        argo_run_id='{{workflow.labels.pipeline/runid}}',
        model=model_train_task.outputs['model_pickle'],
        psi_artifact=train_monitoring_task.outputs["psi_artifact"])
    

    upload_task = upload_model_to_registry(
        project_id=project_id,
        model_name=model_name,
        model_registry_location=model_registry_location,
        serving_container_image_location=serving_image_location,
        artifact_uri=copy_artifacts_to_storage_task.output)\
            .set_display_name('upload_model_to_registry')

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

