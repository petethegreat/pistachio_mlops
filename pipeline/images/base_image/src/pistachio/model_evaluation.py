"""model_evaluation.py
functions around evaluating a trained classifier
"""

import logging
import pickle
import sys

from typing import List, Dict, Callable, Tuple
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import numpy as np



from xgboost import XGBClassifier

def get_evaluation_metrics(predicted_probs, predicted_classes, actual_classes, prefix=None):
    """evaluate results"""
    results = {}
    prefix = '' if prefix is None else prefix
    results[f"{prefix}roc_auc_score"] = roc_auc_score(actual_classes, predicted_probs)
    results[f"{prefix}precision_score"] = precision_score(actual_classes, predicted_classes)
    results[f"{prefix}recall_score"] = recall_score(actual_classes, predicted_classes)
    results[f"{prefix}f1_score"] = f1_score(actual_classes, predicted_classes)
    results[f"{prefix}accuracy_score"] = accuracy_score(actual_classes, predicted_classes)
    return results

def get_roc_results(predicted_probs: List[float], actual_classes: List[float]) -> Tuple[List[float],List[float],List[float]]:
    """get roc curve definition

    Args:
        predicted_probs (List[float]): predicted probabilities
        actual_classes (List[float]): actual binary labels

    Returns:
        Tuple[List,List,List]: fpr, tpr, thresholds
    """
    fpr, tpr, thresholds = roc_curve(actual_classes, predicted_probs)
    if thresholds[0] == float('inf'):
        thresholds[0] = sys.float_info.max

    return fpr, tpr, thresholds

#################################################################

def plot_roc_curve(fpr, tpr, thresholds, title: str="ROC curve", xlabel='False Positive Rate', ylabel: str='True Positive Rate') -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """_summary_

    Args:
        fpr (_type_): _description_
        tpr (_type_): _description_
        thresholds (_type_): _description_
        title (str, optional): _description_. Defaults to "ROC curve".
        xlabel (str, optional): _description_. Defaults to 'False Positive Rate'.
        ylabel (str, optional): _description_. Defaults to 'True Positive Rate'.

    Returns:
        Tuple[mpl.Figure, mpl.Axis]: _description_
    """
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.plot(fpr, tpr, color=sns.xkcd_rgb['blurple'], label='roc curve')
    ax.plot([0.0, 1.0],[0.0, 1.0], color=sns.xkcd_rgb['merlot'], linestyle='--', label='random')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    # fig.show()
    return fig, ax
#################################################################

def plot_feature_importances(model: XGBClassifier, title: str = 'feature importances'):
    feature_importances = [(k,v) for k,v in model.get_booster().get_score(importance_type='gain').items()]
    feature_importances.sort(key = lambda x: x[1], reverse=True)
    feats, importances = zip(*feature_importances)
    cmap  = sns.color_palette("viridis", as_cmap=True)
    norm = mpl.colors.Normalize(vmin=importances[-1], vmax=importances[0])
    colours = [cmap(norm(x)) for x in importances]

    fig = plt.figure()
    ax = fig.add_axes([0.2,0.1,0.8,0.8])
    ypos = np.arange(len(importances),0,-1)
    # sns.xkcd_rgb['blurple']
    ax.barh(ypos, importances, color=colours)
    ax.set_yticks(ypos)
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel('feature importance (gain)')
    return fig, ax
#################################################################

def make_roc_plot(predicted_probs, actual_classes, title: str="ROC curve", xlabel='False Positive Rate',ylabel: str='True Positive Rate'):
    """make a roc curve"""
    fpr, tpr, _ = roc_curve(actual_classes, predicted_probs)
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.plot(fpr, tpr, color=sns.xkcd_rgb['blurple'], label='roc curve')
    ax.plot([0.0, 1.0],[0.0, 1.0], color=sns.xkcd_rgb['merlot'], linestyle='--', label='random')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    # fig.show()
    return fig, ax
#################################################################

def get_confusion_matrix(predicted_classes, actual_classes, normalise=False):
    """get confusion matrix
    computes confusion matrix for binary classification

    Args:
        predicted_classes (_type_): _description_
        actual_classes (_type_): _description_
        normalise (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    matrix = confusion_matrix(actual_classes,predicted_classes, normalize=normalise)
    return matrix
#################################################################

def make_confusion_matrix_plot(
    predicted_classes,
    actual_classes,
    title:str = 'confusion matrix',
    xlabel: str='predicted class',
    ylabel: str='actual class',
    class_names: List[str] = None,
    normalise:str=None
    ):
    """ generate confusion matrix plot"""
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.grid(False)
    # cmap = sns.color_palette("magma_r", as_cmap=True)
    cmap = sns.light_palette("indigo", as_cmap=True)

    # cmap = 'viridis'
    
    matrix = confusion_matrix(actual_classes,predicted_classes, normalize=normalise)
    ax.imshow(matrix, cmap=cmap)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(i,j,f'{matrix[i,j]}')
    # ax.plot([0.0, 1.0],[0.0, 1.0], color=sns.xkcd_rgb['merlot'], linestyle='--', label='random')
    labels = class_names if class_names else ['0','1']

    ax.set_xlim([-0.5, matrix.shape[0]- 0.5])
    ax.set_ylim([matrix.shape[0]- 0.5, -0.5])
    ax.set_xticks(np.arange(matrix.shape[0]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # ax.legend()
    # fig.show()
    return fig, ax
#################################################################

def make_precision_recall_plot(predicted_probs, actual_classes, title: str="ROC curve", xlabel='False Positive Rate',ylabel: str='True Positive Rate',
                              positive_rate:float=None):
    """make a roc curve"""
    precision, recall, _ = precision_recall_curve(actual_classes, predicted_probs)
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    classifier_average_precision = average_precision_score(actual_classes, predicted_probs)
    ax.plot(recall, precision, color=sns.xkcd_rgb['blurple'], label=f'precision recall curve (average precision = {classifier_average_precision:0.3f}')
    if positive_rate:
        ax.plot([0.0, 1.0],[positive_rate, positive_rate], color=sns.xkcd_rgb['merlot'], linestyle='--', label=f'positive response rate = {positive_rate:0.3f}')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend()
    # fig.show()
    return fig, ax

