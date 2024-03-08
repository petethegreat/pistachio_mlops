#!/usr/bin/env python 
"""batch_prediction_pipeline.py
pipeline definition. render component definitions from templates in components directory.
define pipeline from components.
"""

from kfp import dsl
from kfp import compiler
from kfp.registry import RegistryClient

from google_cloud_pipeline_components.types.artifact_types import VertexModel
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.batch_predict_job import ModelBatchPredictOp

# from container_components import hyperparameter_tuning  load_data, preprocess_data, validate_data, train_monitoring
# from container_components import train_final_model, evaluate_trained_model, infer_monitoring
from components import sample_data, validate_data, preprocess_data, infer_monitoring
from components import psi_result_logging, get_model_artifacts_from_registry, prepare_csv_op

import yaml

CONFIG_FILE_PATH = '../config/config.yaml'

with open(CONFIG_FILE_PATH,'r') as config_file:
    CONFIG = yaml.safe_load(config_file)

bucket_name = CONFIG.get('gcs_bucket','the_gcs_bucket')

pipeline_root = f'gs://{bucket_name}/pistachio_prediction_root'

pipeline_name = CONFIG.get('training_pipeline_name','the_pipeline_name')
schema_file_path = f"/gcs/{bucket_name}/pipeline_resources/pistachio_schema.json"
arff_file_path = f"/gcs/{bucket_name}/pipeline_resources/Pistachio_16_Features_Dataset.arff"
project_id = CONFIG.get('project_id', 'the_project_id')

# output_csv_prefix=f'gs://{bucket_name}/pistachio_prediction_output'
# input_csv_prefix=f'gs://{bucket_name}/pistachio_prediction_input'
input_csv_path = CONFIG.get('prediction_input_path','pistachio_prediction_input')
output_csv_path = CONFIG.get('prediction_output_path','pistachio_prediction_output')


# name to use in registry
model_name = CONFIG.get('model_registry_name','pistachio_classifier')
model_registry_location = CONFIG.get('model_registry_location','the_gcp_location')
# need this for the model upload op
artifact_registry = CONFIG.get('artifact_registry', 'the_artifact_registry')
base_image_name = CONFIG.get('base_image_name', 'the_base_image:0.0.0')
base_image_location = f'{artifact_registry}/{base_image_name}'
serving_image_name = CONFIG.get('serving_image_name', 'the_serving_image:0.0.0')
serving_image_location = f'{artifact_registry}/{serving_image_name}'


@dsl.pipeline(
    name='pistachio_batch_prediction_pipeline',
    description='pipeline for batch inference of pistachio classifier',
    pipeline_root=pipeline_root)
def pistachio_prediction_pipeline(
    project_id: str=project_id,
    sample_seed: int=47,
    sample_records: int=1000
    ):
    """prediction pipeline


    """
    

    # validate
    # preprocess 
    # monitor 
    # inference 
    # move output dataset to a bucket location - export artifact to csv

    # load a sample of data to run inference on
    sample_data_task = sample_data(
        input_file_path=arff_file_path,
        sample_seed=sample_seed,
        sample_records=sample_records)\
        .set_display_name('sample dataset')
    
    # validate data
    validate_inference_data_task = validate_data(
        input_file=sample_data_task.outputs['output_sample'],
        schema_file_path=schema_file_path)\
        .set_display_name('validate sample data')

    preprocess_sample_data_task = preprocess_data(
        input_file=sample_data_task.outputs['output_sample'])\
        .after(validate_inference_data_task)\
        .set_display_name('preprocess sample data')

    get_model_task = get_model_artifacts_from_registry(
        project_id=project_id,
        model_name=model_name,
        model_registry_location=model_registry_location)\
            .set_caching_options(enable_caching=False)

    infer_monitor_task = infer_monitoring(
        inference_data=preprocess_sample_data_task.outputs["output_file"],
        psi_artifact=get_model_task.outputs["psi_artifact"])\
        .set_display_name('sample data PSI monitoring')

    # convert parquet to csv for input
    # define gcs path for output - add jobid to prefix - must be done in component
    prepare_csv_task = prepare_csv_op(
        storage_bucket=bucket_name,
        input_parquet_data=preprocess_sample_data_task.outputs["output_file"],
        input_gcs_csv_path=f'{input_csv_path}/{sample_seed}/{dsl.PIPELINE_JOB_ID_PLACEHOLDER}/pistachio_input.csv',
        #output_gcs_csv_path=f'{output_csv_path}/{sample_seed}/{dsl.PIPELINE_JOB_ID_PLACEHOLDER}/',
    )
    
    batch_predict_task = ModelBatchPredictOp(
        project=project_id,
        location=model_registry_location,
        job_display_name='pistachio_classifier_batch_prediction',
        model=get_model_task.outputs['model_artifact'],
        machine_type='e2-standard-2',
        gcs_source_uris=[f'gs://{bucket_name}/{input_csv_path}/{sample_seed}/{dsl.PIPELINE_JOB_ID_PLACEHOLDER}/pistachio_input.csv'],
        instances_format='csv',
        gcs_destination_output_uri_prefix=f'gs://{bucket_name}/{output_csv_path}/{sample_seed}/{dsl.PIPELINE_JOB_ID_PLACEHOLDER}/'
        )\
        .after(prepare_csv_task)\
        .set_display_name('batch prediction task')

    # This works
    # import the artifact
    # https://cloud.google.com/vertex-ai/docs/pipelines/use-components#use_an_importer_node

    # VertexModel metadata
    # https://google-cloud-pipeline-components.readthedocs.io/en/google-cloud-pipeline-components-2.8.0/api/artifact_types.html#google_cloud_pipeline_components.types.artifact_types.VertexModel
    # https://github.com/kubeflow/pipelines/blob/master/components/google-cloud/google_cloud_pipeline_components/types/artifact_types.py#L48

    # model_uri = f'https://https://{model_registry_location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/models/{model}'\
    

# need to create the artifact in the component, not import
# component outputs can't be used here.
    # import_model_task =  dsl.importer(
    #   artifact_uri=get_model_task.output['model_uri'], # this is defined in component, not contained in content from model get
    #   artifact_class=artifact_types.VertexModel,
    #   metadata={
    #       'resourceName': get_model_task.output['name']
    #   }
    # )
    


#     # import the model as an artifact and feed to batchpredictionop.
#     https://google-cloud-pipeline-components.readthedocs.io/en/google-cloud-pipeline-components-2.8.0/api/artifact_types.html

    
    
# https://google-cloud-pipeline-components.readthedocs.io/en/google-cloud-pipeline-components-2.8.0/api/v1/batch_predict_job.html#v1.batch_predict_job.ModelBatchPredictOp
#     model: dsl.Input[google.VertexModel] = None, 
#     batch_predict_task = ModelBatchPredictOp(
#         job_display_name='pistachio_batch_prediction_task',
#         model=

#     )




    

    

    # https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform_v1.services.model_service.ModelServiceClient#google_cloud_aiplatform_v1_services_model_service_ModelServiceClient_get_model
    # get the model data
    # https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform_v1.types.Model
    # can get metadata - if it's specified on upload
    # can get pipeline job and training job - if specified, but these might only be if trainied using a customjob
    # artifact_uri - this is what we want

    # can use V1 version
    # https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform_v1.services.model_service.ModelServiceClient#google_cloud_aiplatform_v1_services_model_service_ModelServiceClient_get_model
    # https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform_v1.types.Model
    # this seems to be borked - getting some sort of rpc error.

    # have a get model component that looks up the model name and gets artifact_uri and whatever details are needed for batch prediction
    # https://google-cloud-pipeline-components.readthedocs.io/en/google-cloud-pipeline-components-2.8.0/api/v1/batch_predict_job.html#v1.batch_predict_job.ModelBatchPredictOp

    # THIS
    # https://cloud.google.com/vertex-ai/docs/pipelines/use-components#create_an_ml_artifact
    # use an importer node to create an artifact of type VertexModel
    






pipeline_output_path = './pipeline_artifact/pistaciho_prediction_pipeline.yaml'   
compiler.Compiler().compile(pistachio_prediction_pipeline, package_path=pipeline_output_path)
print(f"pipeline compiled: {pipeline_output_path}")
# upload
pipeline_registry = CONFIG.get('pipeline_registry')
if pipeline_registry:
        
    client = RegistryClient(host=pipeline_registry)
    template_name, version_name = client.upload_pipeline(
    file_name=pipeline_output_path,
    tags=["v1", "latest"],
    extra_headers={"description":"pistachio prediction pipeline artifact"})
    print(f"uploaded pipeline to registry {pipeline_registry}")
    print(f"template_name: {template_name}")
    print(f"version_name: {version_name}")


