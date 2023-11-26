"""container_components.py
components defined using kfp container_spec
"""

from kfp import dsl
from kfp.dsl import Dataset, Input, Output, InputPath, OutputPath, Artifact, Markdown, Metrics, Model, SlicedClassificationMetrics, ClassificationMetrics
from typing import List, Dict
from kfp.dsl import ConcatPlaceholder
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
    # output_train.path = ConcatPlaceholder([output_train.path, '.pqt'])
    # output_test.path = ConcatPlaceholder([output_test.path, '.pqt'])
    # This does not work - the data is written witha .pqt extension, 
    # but the output_train.path and output_test.path are not updated, or the update is not reflected in the rest of the pipeline


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

    # output_file.path = ConcatPlaceholder([output_file.path, '.pqt'])
    # feature_list.path = ConcatPlaceholder([feature_list.path, '.json'])

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
    # optimal_parameters_json: OutputPath(Dict), # this works. But the containers aren't set up to take a string of json as an argument, rather a filename containing json.
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

@dsl.container_component
def train_final_model(
    preprocessed_train_data: Input[Dataset],
    optimal_parameters_json: Input[Artifact],
    model_pickle: Output[Model],
    featurelist_json: Input[Artifact],
    ) -> dsl.ContainerSpec:
    """model training component

    trains a model and saves it as an artifact (pickle file) based on the parameters obtained from tuning

    Args:
        preprocessed_train_data (Input[Dataset]): preprocessed training data
        optimal_parameters_json (Input[Artifact]): parameters to use for the final model
        model_pickle (Output[Model]): trained model artifact
        featurelist_json (Input[Artifact]): list of features to be used

    Returns:
        dsl.ContainerSpec: containerspec defining this component
    """

    return dsl.ContainerSpec(
        image=base_image_location,
        command=['./train_model.py'],
        args=[
            preprocessed_train_data.path,
            optimal_parameters_json.path,
            model_pickle.path,
            featurelist_json.path]
        )
#############################################################################


@dsl.container_component
def evaluate_trained_model(
    dataset: Input[Dataset],
    model_pickle: Input[Model],
    featurelist_json: Input[Artifact],
    metric_results_json: Output[Artifact],
    feature_importance_plot_png: Output[Artifact],
    roc_curve_plot_png: Output[Artifact],
    metric_prefix: str='metric_',
    dataset_desc: str='dataset'
    ) -> dsl.ContainerSpec:
    """evaluate trained model on specified dataset

    Args:
        dataset (Input[Dataset]): dataset to be used for model inference
        model_pickle (Input[Model]): pickle file containing pistachio XGBClassifier
        featurelist_json (Input[Artifact]): list of features
        metric_results_json (Output[Artifact]): location where evaluation metrics will be written (as json)
        feature_importance_plot_png (Output[Artifact]): path where feature importance plot will be written as png
        roc_curve_plot_png (Output[Artifact]): path where roc curve plot will be written as png
        metric_prefix (str, optional): string added as a prefix to metric keys. Defaults to 'metric_'.
        dataset_desc (str, optional): dataset description, used in plot titles. Defaults to 'dataset'.

    Returns:
        dsl.ContainerSpec: _description_
    """

    # metric_results_json.path = ConcatPlaceholder([metric_results_json.path, '.json'])
    # feature_importance_plot_png.path = ConcatPlaceholder([feature_importance_plot_png.path, '.png'])
    # roc_curve_plot_png.path = ConcatPlaceholder([roc_curve_plot_png.path, '.png'])
 
    return dsl.ContainerSpec(
        image=base_image_location,
        command=['./evaluate_model.py'],
        args=[
            dataset.path,
            model_pickle.path,
            featurelist_json.path,
            metric_results_json.path,
            feature_importance_plot_png.path,
            roc_curve_plot_png.path,
            "--metric_prefix",
            metric_prefix,
            "--dataset_desc",
            dataset_desc]
        )
