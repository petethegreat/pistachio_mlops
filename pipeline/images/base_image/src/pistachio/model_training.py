"""model_training.py
functions around training/tuning classifier
"""

import logging
import pickle

from typing import List, Dict, Callable, Tuple

import numpy as np
import pandas as pd
import sklearn 
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold, cross_validate


logger = logging.getLogger(__name__)

def cross_validate_estimator(
    train_X: pd.DataFrame,
    train_Y: pd.DataFrame,
    clf: sklearn.base.BaseEstimator,
    metrics: Dict[str, Callable],
    n_folds: int=5,
    cv_seed:int=23,
    n_jobs: int=2):
    """for a given set of model parameters, use cross validation to evaluate model performance"""
    
    # generate cv_folds
    cv_folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_seed)
    results = cross_validate(clf, train_X, train_Y, cv=cv_folds, scoring=metrics, n_jobs=n_jobs)

    return results

def cast_integer_params(params: Dict, integer_params: List[str]):
    """cast floats in param values to integers"""
    for x in integer_params:
        if x in params:
            params[x] = int(params[x])
    return params

# metrics = {
#     "roc_auc": make_scorer(roc_auc_score),
#     "precision": make_scorer(precision_score),
#     "recall": make_scorer(recall_score),
#     "f1_score": make_scorer(f1_score)
# }

def optimise_tune(
    train_x: pd.DataFrame,
    train_y: pd.DataFrame,
    pbounds: Dict,
    fixed_parameters: Dict[str, int | float | str],
    integer_parameters: List[str],
    metrics: Dict[str, Callable],
    cv_seed: int,
    n_folds: int=5,
    opt_n_init: int=10,
    opt_n_iter: int=20,
    opt_random_seed: int=42
) -> Tuple[Dict, List[Dict]]:
    """use Bayesian optimisation to search for optimal model hyperparameters

    Args:
        train_x (pd.DataFrame): Training data - dataframe of features
        train_y (pd.DataFrame): Training data - dataframe of labels (targets). Should be integers (1 or 0) - binary classification
        pbounds (Dict): Parameter boundaries. Defines the hyperspace to be searched.
        fixed_parameters (Dict): Static parameters that should be passed to the classifier - These will not be varied across trials
        integer_parameters (List[str]): List of integer valued parameters. BayesOpt assumes parameters are continuous, so there is some special handling required for integers.
        metrics (Dict[str, Callable]): Metrics that will be evaluated within each trial. This is a dictionary mapping metric names to sklearn scorers (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html)
        cv_seed (int): seed used to handle cv fold splitting
        n_folds (int, optional): number of folds to use within each trial. Defaults to 5.
        opt_n_init (int, optional): initial number of initialisation trials - initial data used to feed optimisation. These are randomly chosen. Defaults to 10.
        opt_n_iter (int, optional): numer of optimisation iterations to run. BayesOpt will use what it knows to search the most promising areas of parameter hyperspace. Defaults to 20.
        opt_random_seed (int, optional): _description_. Defaults to 42.

    Returns:
        _type_: _description_
    """

    # initialise list to hold (detailed) experiment results
    trials = []
    # discrete parameters need to be handled specially in bayesopt (explicitly cast to int)


    # function to run a trial - evaluate a given set of searchable parameters
    def run_trial(**probe_params):
        params = {**probe_params, **fixed_parameters}
        params = cast_integer_params(params, integer_parameters)


        # set up the XGBclassifier
        clf = XGBClassifier(objective='binary:logistic', eval_metric='auc', **params)

        # train/evaluate model through cross validation
        results = cross_validate_estimator(train_x, train_y, clf, metrics, cv_seed=cv_seed, n_jobs=2)

        # aggregate metrics over cv folds - gather mean and std deviation of each metric
        agged_results = {k:(np.mean(v), np.std(v)) for k,v in results.items()}

        # take the final score - the objective to be used for bayes_opt, as the lower bound of mean roc_auc (mean roc_auc minus error in mean roc_auc)
        # This penalises cases where the mean might be high, but where there is more variation across folds (more uncertainty in how the model may generalise).
        final_score = agged_results['test_roc_auc'][0] - agged_results['test_roc_auc'][1]/np.sqrt(n_folds)

        # append all the metrics to the trial result.
        trials.append( {"final_score": final_score, 'params': params, "results": agged_results})
        logger.info(f'{len(trials)} trials done')
        return final_score

    optimizer = BayesianOptimization(
        f = run_trial,
        random_state=opt_random_seed,
        pbounds=pbounds,
        verbose=2)

    optimizer.maximize(init_points=opt_n_init, n_iter=opt_n_iter)

    logger.info(f"best_result: {optimizer.max}")
    return optimizer.max, trials
##############################################

def train_model(
    train_x: pd.DataFrame,
    train_y: pd.DataFrame,
    params: Dict
): 
    """Train a model on entire train set"""
    clf = XGBClassifier(objective='binary:logistic', eval_metric='auc', **params)

    model = clf.fit(train_x, train_y)
    return model
##############################################

def save_model_to_pickle(
        model: XGBClassifier,
        model_pickle_output: str) -> None:
    """write a trained model to pickle file

    Args:
        model (XGBClassifier): model to be saved
        model_pickle_output (str): output pickle file location
    """
    with open(model_pickle_output, 'wb') as outfile:
        pickle.dump(model, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"wrote model to {model_pickle_output}")
##############################################

def read_model_from_pickle(
        model_pickle_input: str) -> XGBClassifier:
    """_summary_

    Args:
        model_pickle_input (str): _description_

    Returns:
        XGBClassifier: _description_
    """
    
    with open(model_pickle_input, 'rb') as infile:
        model = pickle.load(infile)
    logger.info(f"loaded model from {model_pickle_input}")
    return model
##############################################
