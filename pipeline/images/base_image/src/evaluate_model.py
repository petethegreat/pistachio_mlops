#!/usr/bin/env python
"""evaluate_model.py

take a dataset and model artifact, generate metrics and plots
"""



from argparse import ArgumentParser
from pistachio.data_handling import load_parquet_file, read_from_json, write_to_json
from pistachio.model_evaluation import get_evaluation_metrics, plot_feature_importances, get_roc_results, plot_roc_curve
from pistachio.model_evaluation import get_confusion_matrix, make_confusion_matrix_plot
from pistachio.utils import ensure_directory_exists
from pistachio.model_training import read_model_from_pickle
import seaborn as sns
from typing import List

import pickle
import logging
import sys

## logging
logger = logging.getLogger('pistachio')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

sns.set()

#########################################################

def evaluate_model(
    dataset_path: str,
    model_pickle_path: str,
    featurelist_json: str,
    metric_results_json: str,
    feature_importance_plot_png: str,
    roc_curve_plot_png: str,
    metric_prefix: str,
    dataset_desc: str,
    ):
    """evaluate model performance on a dataset

    Args:
        dataset_path (str): _description_
        model_pickle_path (str): _description_
        featurelist_json (str): _description_
        metric_prefix (str): _description_
        dataset_desc (str): _description_
    """
    features = read_from_json(featurelist_json)
    evaluate_model_features(
        dataset_path=dataset_path,
        model_pickle_path=model_pickle_path,
        features=features,
        metric_results_json=metric_results_json,
        feature_importance_plot_png=feature_importance_plot_png,
        roc_curve_plot_png=roc_curve_plot_png,
        metric_prefix=metric_prefix,
        dataset_desc=dataset_desc)
#########################################################

def evaluate_model_features(
    dataset_path: str,
    model_pickle_path: str,
    features: List[str],
    metric_results_json: str,
    feature_importance_plot_png: str,
    roc_curve_plot_png: str,
    confusion_matrix_plot_png: str,
    pr_plot_png: str,
    probability_plot_png: str,
    metric_prefix: str,
    dataset_desc: str,
    n_bins_prob: int=50
    ):
    """evaluate model performance on a dataset

    Args:
        dataset_path (str): path to dataset
        model_pickle_path (str): path to model artifact
        features (List[str]): features needed by model
        metric_results_json (str): path for evaluation metrics output json
        feature_importance_plot_png (str): output path for feature importance plot
        roc_curve_plot_png (str): output path for ROC curve plot
        confusion_matrix_plot_png (str): output path for confusion matrix plot
        pr_plot_png (str): output path for precision-recall plot
        probability_plot_png (str): output path for probability calibration plot
        metric_prefix (str): prefix for metric identifiers
        dataset_desc (str): string describing dataset (e.g. train/validation/test)
        n_bins_prob (int): number of bins to use in probability calibration plot
    """


    dataset = load_parquet_file(dataset_path)
    model = read_model_from_pickle(model_pickle_path)

    for path in [metric_results_json, feature_importance_plot_png, roc_curve_plot_png, 
        confusion_matrix_plot_png, pr_plot_png, probability_plot_png]:
        ensure_directory_exists(path)

    dataset_features = dataset[features]
    dataset_binary_labels = dataset.Target.values
    dataset_class_labels = dataset.Class

    class_lookup = dataset.Class.cat.categories

    predicted_probabilities = model.predict_proba(dataset_features)[:,1]
    predicted_binary_labels = model.predict(dataset_features)
    predicted_class_labels = [class_lookup[x] for x in predicted_binary_labels]

    # compute classification metrics
    metrics = {}
    metrics['metrics'] = get_evaluation_metrics(
        predicted_probabilities,
        predicted_binary_labels,
        dataset_binary_labels,
        prefix=metric_prefix)
    
    fpr, tpr, thresholds = get_roc_results(predicted_probabilities, dataset_binary_labels)
    metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thresholds.tolist()}

    conf_matrix = get_confusion_matrix(predicted_binary_labels, dataset_binary_labels)
    metrics['confusion_matrix'] = conf_matrix

    write_to_json(metrics, metric_results_json)

    # feature importance plot
    fig, ax = plot_feature_importances(model, title=f'Model Feature importances for Pistachio classifier')
    fig.savefig(feature_importance_plot_png, format='png')

    # roc curve
    fig, ax = plot_roc_curve(fpr, tpr, thresholds, title=f'ROC curve - {dataset_desc}')
    fig.savefig(roc_curve_plot_png, format='png')

    # confusion matrix
    class_names = [class_lookup[0], class_lookup[1]]
    fig, ax = make_confusion_matrix_plot(
        predicted_binary_labels, dataset_binary_labels,
        title=f'confusion matrix {dataset_desc}',
        class_names=class_names) 
    fig.savefig(confusion_matrix_plot_png, format='png')

    # PR plot
    dataset_positive_response_rate = np.sum(dataset_binary_labels)/len(dataset_binary_labels)
    fig, ax = make_precision_recall_plot(
        predicted_probabilities,
        dataset_binary_labels,
        positive_rate=dataset_positive_response_rate,
        title=f'Precision Recall Plot - {dataset_desc}')
    fig.savefig(pr_plot_png, format='png')

    # probability calibration

    fig,ax = make_prob_calibration_plot(
        predicted_probabilities,
        dataset_binary_labels,
        n_bins=n_bins_prob,
        title=f'probability calibration - {dataset_desc}')
    fig.savefig(probability_plot_png, format=png)

    logger.info(f'model evaluation on {dataset_desc} done')



#########################################################
def main():

    parser = ArgumentParser(
        description="evaluate model on a dataset and generate performance metrics and plots"
    )
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('model_pickle_path', type=str)
    parser.add_argument('featurelist_json', type=str)
    parser.add_argument('metric_results_json', type=str)
    parser.add_argument('feature_importance_plot_png', type=str)
    parser.add_argument('roc_curve_plot_png', type=str)
    parser.add_argument('confusion_matrix_plot_png', type=str)
    parser.add_argument('pr_plot_png', type=str)
    parser.add_argument('probability_plot_png', type=str)

    parser.add_argument('--metric_prefix', type=str, default='dataset_metric_')
    parser.add_argument('--dataset_desc', type=str, default='dataset')

    args = parser.parse_args()

    evaluate_model(
        dataset_path=args.dataset_path,
        model_pickle_path=args.model_pickle_path,
        featurelist_json=args.featurelist_json,
        metric_results_json=args.metric_results_json,
        feature_importance_plot_png=args.feature_importance_plot_png,
        roc_curve_plot_png=args.roc_curve_plot_png,
        confusion_matrix_plot_png=confusion_matrix_plot_png,
        pr_plot_png=pr_plot_png,
        probability_plot_png=probability_plot_png,
        metric_prefix=args.metric_prefix,
        dataset_desc=args.dataset_desc)
  

if __name__ == "__main__":
    main()