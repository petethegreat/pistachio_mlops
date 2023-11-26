"""model_evaluation.py
functions around evaluating a trained classifier
"""

import logging
import pickle

from typing import List, Dict, Callable, Tuple

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

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
