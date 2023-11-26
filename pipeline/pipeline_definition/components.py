"""components.py
components defined using kfp component decorator
lightweight components that have a base image specified.
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


#############################################################################
@dsl.component(base_image=base_image_location)
def load_data2(input_file_path: str,
    output_train: Output[Dataset],
    output_test: Output[Dataset],
    split_seed: int=37,
    test_fraction: float=0.2,
    label_column: str='Class'
    ):
    """load_data2
    lightweight component to load data from arff file and write to parquet

    Args:
        input_file_path (str): _description_
        output_train (Output[Dataset]): path train data will be written to
        output_test (Output[Dataset]): path test data will be written to 
        split_seed (int, optional): seed used when carrying out train/test split. Defaults to 37.
        test_fraction (float, optional): fraction of data to be allocated to test split. Defaults to 0.2.
        label_column (str, optional): Label column in that dataset - used to stratify the splitting. Defaults to 'Class'.
    """

    from load_data import load_and_split_data
    # load_data.py is the python file in the image - the container components run this as an entrypoint

    output_train.path = output_train.path + '.pqt'
    output_test.path = output_test.path + '.pqt'

    load_and_split_data(
        input_file_path=input_file_path,
        output_train_file_path=output_train.path,
        output_test_file_path=output_train.path,
        split_seed=split_seed,
        test_fraction=test_fraction,
        label_column=label_column
    )

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
    fpr = [ 0.0, 0.0, 0.0, 1.0]
    tpr = [0.0, 0.5, 1.0, 1.0]
    thresholds = [sys.float_info.max, 0.99, 0.8, 0.01] # infinity is an issue
    # test_evaluation_metrics.log_roc_curve(fpr, tpr, thresholds)
    train_evaluation_metrics.log_roc_curve(fpr, tpr, thresholds)

    
    
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


