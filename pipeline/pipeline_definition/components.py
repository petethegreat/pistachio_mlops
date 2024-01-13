"""components.py
components defined using kfp component decorator
lightweight components that have a base image specified.
"""

from kfp import dsl
from kfp.dsl import Dataset, Input, Output, InputPath, OutputPath, Artifact, Markdown, Metrics, Model, SlicedClassificationMetrics, ClassificationMetrics
from typing import List, Dict, Tuple
from kfp.dsl import ConcatPlaceholder
import yaml

CONFIG_FILE_PATH = '../config/config.yaml'

with open(CONFIG_FILE_PATH,'r') as config_file:
    CONFIG = yaml.safe_load(config_file)

artifact_registry = CONFIG.get('artifact_registry', 'the_artifact_registry')
base_image_name = CONFIG.get('base_image_name', 'the_base_image:0.0.0')

base_image_location = f'{artifact_registry}/{base_image_name}'
vertex_image_name = CONFIG.get('vertex_image_name','the_vertex_image:0.0.0')
vertex_image_location = f'{artifact_registry}/{vertex_image_name}'


#############################################################################
@dsl.component(base_image=base_image_location)
def load_data(input_file_path: str,
    output_train: Output[Dataset],
    output_test: Output[Dataset],
    split_seed: int=37,
    test_fraction: float=0.2,
    label_column: str='Class'
    )-> None:
    """load_data
    component to load data from arff file and write to parquet

    Args:
        input_file_path (str): _description_
        output_train (Output[Dataset]): path train data will be written to
        output_test (Output[Dataset]): path test data will be written to 
        split_seed (int, optional): seed used when carrying out train/test split. Defaults to 37.
        test_fraction (float, optional): fraction of data to be allocated to test split. Defaults to 0.2.
        label_column (str, optional): Label column in that dataset - used to stratify the splitting. Defaults to 'Class'.

    Returns:
        None
    """

    from load_data import load_and_split_data
    # load_data.py is the python file in the image - the container components run this as an entrypoint

    output_train.path = output_train.path + '.pqt'
    output_test.path = output_test.path + '.pqt'

    load_and_split_data(
        input_file_path=input_file_path,
        output_train_file_path=output_train.path,
        output_test_file_path=output_test.path,
        split_seed=split_seed,
        test_fraction=test_fraction,
        label_column=label_column
    )

#############################################################################

@dsl.component(base_image=base_image_location)
def validate_data(
    input_file: Input[Dataset],
    schema_file_path: str
    )-> None:
    """validate_data component

    Args:
        input_file (Input[Dataset]): path to input dataset to be validated
        schema_file_path (str): pandera schema file to use for validation

    Returns:
        None
    """
    from validate_data import validate_data
    validate_data(input_file.path, schema_file_path)
#############################################################################
@dsl.component(base_image=base_image_location)
def preprocess_data(
    input_file: Input[Dataset],
    output_file: Output[Dataset],
    )-> None:
    """preprocess_data component

    Args:
        input_file (Input[Dataset]): path to raw data to be preprocessed
        output_file (Output[Dataset]): path where preprocessed data will be written 
        feature_list (Output[Artifact]): path to where list of feature columns will be written as json

    Returns:
        None
    """   
    from preprocess_data import preprocess_data_features

    output_file.path = output_file.path + '.pqt'

    features = preprocess_data_features( input_file.path, output_file.path)
    output_file.metadata['features'] = features
#############################################################################

@dsl.component(base_image=base_image_location)
def train_monitoring(
    train_data: Input[Dataset],
    psi_artifact: Output[Artifact]
    ) -> None:
    """train_monitoring component

    Args:
        train_data (Input[Dataset]): preprocessed training data
        psi_artifact (Output[Artifact]): PSI artifact containing trained PSIMetrics object

    Returns:
        None
    """

    from train_monitoring import fit_psi
    
    psi_artifact.path = psi_artifact.path + '.pkl'
    fit_psi(train_data.path, psi_artifact.path)
#############################################################################

@dsl.component(base_image=base_image_location)
def infer_monitoring(
    inference_data: Input[Dataset],
    psi_artifact: Input[Artifact],
    psi_results_json: Output[Artifact]
    ) -> None:
    """inference monitoring component
    check for data drift when running inference

    Args:
        inference_data (Input[Dataset]): Dataset to be used for model inference
        psi_artifact (Input[Artifact]): PSI object containing statistics computed at training time
        psi_results_json (Output[Artifact]): PSI results as json file

    Returns:
        None
    """
   
    from infer_monitor import eval_psi
    from pistachio.data_handling import read_from_json

    psi_results_json.path = psi_results_json.path + '.json'

    eval_psi(inference_data.path, psi_artifact.path, psi_results_json.path)
    # attach psi results as metadata
    psi_results = read_from_json(psi_results_json.path)
    inference_data.metadata['psi_evaluation_results'] = psi_results
#############################################################################
# tuples here are not ok
@dsl.component(base_image=base_image_location)
def hyperparameter_tuning(
    preprocessed_train_data: Input[Dataset],
    tuning_results_json: Output[Artifact],
    optimal_parameters_json: Output[Artifact],
    cv_seed: int=43,
    cv_n_folds: int=5,
    opt_n_init: int=10,
    opt_n_iter: int=200,
    opt_random_seed: int=73,
    learning_rate_bounds: List[float]=[0.01, 0.3],
    gamma_bounds: List[float]=[0.0, 0.3],
    min_child_weight_bounds: List[float]=[0.01, 0.07],
    max_depth_bounds: List[int]=[3, 5],
    subsample_bounds: List[float]=[0.7, 0.9],
    reg_alpha_bounds: List[float]=[0.01, 0.1],
    reg_lambda_bounds: List[float]=[0.01, 0.1],
    colsample_bytree_bounds: List[float]=[0.1, 0.5]
    ) -> None:
    """hyperparameter tuning component
    tunes an CGB classifier using bayesopt to search hyperparameter space

    Args:
        preprocessed_train_data (Input[Dataset]): path to preprocessed training dataset (parquet)
        tuning_results_json (Output[Artifact]): output path for tuning results/details (json)
        optimal_parameters_json (Output[Artifact]): output path to best parameter set found (json)
        cv_seed (int, optional): seed used for splitting fold definition in cross validation. Defaults to 43.
        cv_n_folds (int, optional): number of folds for cross validation. Defaults to 5.
        opt_n_init (int, optional): number of initial (random) trials, prior to optimised searching. Defaults to 10.
        opt_n_iter (int, optional): number of search trials to run. Defaults to 200.
        opt_random_seed (int, optional): random seed to be used during search process. Defaults to 73.

    Returns:
        None
    """
    from model_tuning import model_tune_features
    tuning_results_json.path = tuning_results_json.path + '.json'
    optimal_parameters_json.path = optimal_parameters_json.path + '.json'

    pbounds = {
        'learning_rate': (learning_rate_bounds[0], learning_rate_bounds[1]),
        'gamma': (gamma_bounds[0], gamma_bounds[1]),
        'min_child_weight': (min_child_weight_bounds[0], min_child_weight_bounds[1]),
        'max_depth': (max_depth_bounds[0], max_depth_bounds[1]),
        'subsample': (subsample_bounds[0], subsample_bounds[1]),
        'reg_alpha': (reg_alpha_bounds[0], reg_alpha_bounds[1]),
        'reg_lambda': (reg_lambda_bounds[0], reg_lambda_bounds[1]),
        'colsample_bytree': (colsample_bytree_bounds[0], colsample_bytree_bounds[1])
    }

    features = preprocessed_train_data.metadata['features']
    model_tune_features(
        train_file=preprocessed_train_data.path,
        features=features,
        tune_results_json=tuning_results_json.path,
        optimal_parameters_json=optimal_parameters_json.path,
        pbounds=pbounds,
        cv_seed=cv_seed,
        n_folds=cv_n_folds,
        opt_n_init=opt_n_init,
        opt_n_iter=opt_n_iter,
        opt_random_seed=opt_random_seed
    )
#############################################################################

@dsl.component(base_image=base_image_location)
def train_final_model(
    preprocessed_train_data: Input[Dataset],
    optimal_parameters_json: Input[Artifact],
    model_pickle: Output[Model],
    ) -> None:
    """model training component

    trains a model and saves it as an artifact (pickle file) based on the parameters obtained from tuning

    Args:
        preprocessed_train_data (Input[Dataset]): preprocessed training data
        optimal_parameters_json (Input[Artifact]): parameters to use for the final model
        model_pickle (Output[Model]): trained model artifact

    Returns:
        None
    """
    from train_model import train_final_model_features

    model_pickle.path = model_pickle.path + '.pkl'
    features = preprocessed_train_data.metadata['features']

    train_final_model_features(
        training_data_path=preprocessed_train_data.path,
        optimal_parameters_json_path=optimal_parameters_json.path,
        output_model_artifact_path=model_pickle.path,
        features=features
    )
#############################################################################
@dsl.component(base_image=base_image_location)
def evaluate_trained_model(
    dataset: Input[Dataset],
    model_pickle: Input[Model],
    metric_results_json: Output[Artifact],
    feature_importance_plot_png: Output[Artifact],
    roc_curve_plot_png: Output[Artifact],
    confusion_matrix_plot_png: Output[Artifact],
    pr_plot_png: Output[Artifact],
    probability_plot_png: Output[Artifact],
    metric_prefix: str='metric_',
    dataset_desc: str='dataset'
    ) -> None:
    """evaluate trained model on specified dataset

    Args:
        dataset (Input[Dataset]): dataset to be used for model inference
        model_pickle (Input[Model]): pickle file containing pistachio XGBClassifier
        metric_results_json (Output[Artifact]): location where evaluation metrics will be written (as json)
        feature_importance_plot_png (Output[Artifact]): path where feature importance plot will be written as png
        roc_curve_plot_png (Output[Artifact]): path where roc curve plot will be written as png
        metric_prefix (str, optional): string added as a prefix to metric keys. Defaults to 'metric_'.
        dataset_desc (str, optional): dataset description, used in plot titles. Defaults to 'dataset'.

    Returns:
       None
    """
    from evaluate_model import evaluate_model_features
    features = dataset.metadata['features']
    
    metric_results_json.path = metric_results_json.path + '.json'
    feature_importance_plot_png.path = feature_importance_plot_png.path + '.png'
    roc_curve_plot_png.path = roc_curve_plot_png.path + '.png'
    confusion_matrix_plot_png.path = confusion_matrix_plot_png.path + '.png'
    pr_plot_png.path = pr_plot_png.path + '.png'
    probability_plot_png.path = probability_plot_png.path + '.png'

    evaluate_model_features(
    dataset_path=dataset.path,
    model_pickle_path=model_pickle.path,
    features=features,
    metric_results_json=metric_results_json.path,
    feature_importance_plot_png=feature_importance_plot_png.path,
    roc_curve_plot_png=roc_curve_plot_png.path,
    confusion_matrix_plot_png=confusion_matrix_plot_png.path,
    pr_plot_png=pr_plot_png.path,
    probability_plot_png=probability_plot_png.path,
    metric_prefix=metric_prefix,
    dataset_desc=dataset_desc)
#############################################################################

@dsl.component(base_image='python:3.11')
def evaluation_reporting(
    train_evaluation_results_json: Input[Artifact],
    test_evaluation_results_json: Input[Artifact],
    feature_importance_plot_png: Input[Artifact],
    train_roc_curve_plot_png: Input[Artifact],
    test_roc_curve_plot_png: Input[Artifact],
    evaluation_markdown: Output[Markdown],
    # evaluation_metrics: Output[SlicedClassificationMetrics] # SlicedClassificationMetrics is buggy
    train_evaluation_metrics: Output[ClassificationMetrics],
    test_evaluation_metrics: Output[ClassificationMetrics],
    ):
    """evaluation_reporting
        Generate markdown output and log metrics from json files containing evaluation results


    Args:
        train_evalution_results_json (Input[Artifact]): _description_
        test_evaluation_results_json (Input[Artifact]): _description_
        feaure_importance_plot_png (Input[Artifact]): _description_
        evaluation_markdown (Output[Markdown]): _description_
        evaluation_metrics (Output[SlicedClassificationMetrics]): _description_
    """
    
    import json
    import os
    import logging
    import sys

    logger = logging.getLogger('pistachio.evaluation_reporting')
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    evaluation_markdown.path = evaluation_markdown.path + '.md'


    # load the json content
    with open(test_evaluation_results_json.path,'r') as infile:
        test_results = json.load(infile)
    with open(train_evaluation_results_json.path,'r') as infile:
        train_results = json.load(infile)
    
    # setup a string for markdown content
    # include a table header
    markdown_content = f"# Pistachio Classifier Evaluation Results\n\n## Feature Importance\n\n![Feature Importance png]({feature_importance_plot_png.uri})\n\n"
    # Population Stability Index evaluation\n\n{md_note}\n\n" + \
    #     "| Column | Datatype | Missing Values | PSI |\n|--------|----------|----------------|-----|\n"
    
    
    markdown_content += '## Train Set Metrics\n\n| *Metric* | *Value* |\n|--------|--------|\n'

    # Train result metrics
    for k,v in train_results['metrics'].items():
        markdown_content += f'| {k} | {v} |\n'
        # try this
        # train_evaluation_metrics.metadata[k] = v


    markdown_content += f'\n ## Train Set ROC curve\n\n![Train Set ROC curve png]({train_roc_curve_plot_png.uri})\n\n'
    
    markdown_content += '## Test Set Metrics\n\n| *Metric* | *Value* |\n|--------|--------|\n'
    # test result metrics
    for k,v in test_results['metrics'].items():
        markdown_content += f'| {k} | {v} |\n'
        # try this
        # test_evaluation_metrics.metadata[k] = v
    markdown_content += f'\n ## Test Set ROC curve\n\n![Test Set ROC curve png]({test_roc_curve_plot_png.uri})\n\n'
    
    # sliced metrics - this is buggy at present
    # evaluation_metrics._sliced_metrics = {}
    # get roc curve definition

    # test_roc_curve_definition = [
    #     test_results['roc_curve']['thresholds'],
    #     test_results['roc_curve']['tpr'],
    #     test_results['roc_curve']['fpr']]
    
    # evaluation_metrics.load_roc_readings('test', test_roc_curve_definition)
    # hack
    test_results['roc_curve']['thresholds'][0] = 1.0e9

    test_evaluation_metrics.log_roc_curve(
        test_results['roc_curve']['fpr'],
        test_results['roc_curve']['tpr'],
        test_results['roc_curve']['thresholds'])
    
    # train_evaluation_metrics.log_roc_curve(
    #     train_results['roc_curve']['fpr'],
    #     train_results['roc_curve']['tpr'],
    #     train_results['roc_curve']['thresholds'])

    # dummy roc data
    # fpr = [ 0.0, 0.0, 0.0, 1.0]
    # tpr = [0.0, 0.5, 1.0, 1.0]
    # thresholds = [sys.float_info.max, 0.99, 0.8, 0.01] # infinity is an issue
    # test_evaluation_metrics.log_roc_curve(fpr, tpr, thresholds)
    # train_evaluation_metrics.log_roc_curve(fpr, tpr, thresholds)
    train_results['roc_curve']['thresholds'][0] = 1.0e9

    train_evaluation_metrics.log_roc_curve(
        train_results['roc_curve']['fpr'],
        train_results['roc_curve']['tpr'],
        train_results['roc_curve']['thresholds'])



    # write markdown content
    output_dir = os.path.dirname(evaluation_markdown.path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(evaluation_markdown.path,'w') as outfile:
        outfile.write(markdown_content)
        logger.info(f'markdown written to {evaluation_markdown.path}')
    logger.info('done model evaluation reporting')
#############################################################################

#############################################################################
@dsl.component(base_image='python:3.11')
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

    psi_markdown.path = psi_markdown.path + '.md'

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
    markdown_content = f"# PSI results\nPopulation Stability Index evaluation\n\n{md_note}\n\n" + \
        "| Column | Datatype | Missing Values | PSI |\n|--------|----------|----------------|-----|\n"

    # log psi metrics
    for column_name in psi_details.keys():
        the_dtype = psi_details[column_name].get('datatype','unknown')
        n_missing = psi_details[column_name].get('eval_missing',' ')
        psi_value = psi_details[column_name].get('PSI','')
        table_content = f'| {column_name} | {the_dtype} | {n_missing} | {psi_value} |\n'

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

@dsl.component(base_image=vertex_image_location)
def upload_model_to_registry(project_id: str, model_name: str, model_registry_location: str, model: Input[Model]):
    """upload model to vertex ai model registry"""
    # https://google-cloud-pipeline-components.readthedocs.io/en/google-cloud-pipeline-components-2.0.0/api/v1/model.html#v1.model.ModelUploadOp
    # https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/google_cloud_pipeline_components_model_train_upload_deploy.ipynb
    # https://cloud.google.com/vertex-ai/docs/model-registry/import-model#import_a_model_programmatically

    import logging
    import sys
    from google.cloud import aiplatform

    logger = logging.getLogger('pistachio.upload_model_to_registry')
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    aiplatform.init(project=project_id, location=model_registry_location)

    # artifact uri should be a directory, not the specific pickle file
    artifact_uri = model.uri[:rindex('/')+1]

    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=base_image_location,
        serving_container_predict_route='/predict',
        serving_container_health_route='/health',
        # instance_schema_uri=instance_schema_uri,
        # parameters_schema_uri=parameters_schema_uri,
        # prediction_schema_uri=prediction_schema_uri,
        description='model for classifying types of pistachios',
        serving_container_command='./serve_predictions.py',
        # serving_container_args=serving_container_args,
        # serving_container_environment_variables=serving_container_environment_variables,
        # serving_container_ports=serving_container_ports,
        # explanation_metadata=explanation_metadata,
        # explanation_parameters=explanation_parameters,
        sync=True,
    )

    model.wait()

    logger.info(model.display_name)
    logger.info(model.resource_name)






