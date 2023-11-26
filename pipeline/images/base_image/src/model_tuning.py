#!/usr/bin/env python
"""model_tuning.py
code for model tuning component
hyperparameter search for xgboost classifier
"""


from argparse import ArgumentParser
from pistachio.data_handling import load_parquet_file, write_to_json, read_from_json
from pistachio.model_training import optimise_tune, cast_integer_params
from typing import Dict, Tuple
import os

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, make_scorer

import logging
import sys
## logging

logger = logging.getLogger('pistachio')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def model_tune(
    train_file: str,
    features_path: str,
    tune_results_json: str,
    optimal_parameters_json: str,
    pbounds: Dict[str, Tuple[float,float]],
    cv_seed: int=29,
    n_folds: int=5,
    opt_n_init: int=10,
    opt_n_iter: int=50,
    opt_random_seed: int=43
) -> None:
    """model_tune
    run hyperparameter tuning for an XGBClassifier using BayesOpt

    Args:
        train_file (str): path to preprocessed training data file
        features_path (str): path to features list (json)
        tune_results_json (str): path to write tuning results details
        optimal_parameters_json (str): path to write optimal model parameters
        pbounds (Dict[str,tuple[float,float]]): boundaries of the hyperparameter space to be searched
        cv_seed (int, optional): seed used for splitting/defining cross validation folds. Defaults to 29.
        n_folds (int, optional): number of folds to use in cross validation. Defaults to 5.
        opt_n_init (int, optional): number of initial (random/exploration/unguided) trials to conduct. Defaults to 10.
        opt_n_iter (int, optional): number of search/optimisation trials to conduct. Defaults to 50.
        opt_random_seed (int, optional): random seed for optimisation/searching. Defaults to 43.
    """

    logger.info('loading data')
    train_data_df = load_parquet_file(train_file)
    # seperate features and label
    features = read_from_json(features_path)
    train_x = train_data_df[features]
    train_y = train_data_df.Target.values
    
    logger.info('tuning model')

    fixed_parameters = {
        "booster": "gbtree",
        "n_jobs": 1}
    
    integer_parameters = ['max_depth']

    metrics = {
    "roc_auc": make_scorer(roc_auc_score),
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score),
    "f1_score": make_scorer(f1_score)
    }

    best_result, details = optimise_tune(
        train_x,
        train_y,
        pbounds,
        fixed_parameters,
        integer_parameters,
        metrics,
        cv_seed,
        n_folds,
        opt_n_init,
        opt_n_iter,
        opt_random_seed
    )

    logger.info('tuning complete')
    logger.info(f'best result: {best_result}')

    optimal_parameters = {**(best_result['params']),**fixed_parameters}
    optimal_parameters = cast_integer_params(optimal_parameters, integer_parameters)

    # create output directory, if it does not already exist
    for path in [tune_results_json, optimal_parameters_json]:
        output_dir = os.path.dirname(tune_results_json)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    write_to_json(optimal_parameters, optimal_parameters_json)
    logger.info(f'wrote optimal results to {tune_results_json}')

    write_to_json(details, tune_results_json)
    logger.info(f'wrote tuning result details to {tune_results_json}')
################################################################################

def main():
    """do the things"""

    parser = ArgumentParser(
        description="search hyperparameter space to determine optimal hyperparameters for model"
    )
    parser.add_argument('train_file', type=str)
    parser.add_argument('features_json', type=str)

    parser.add_argument('tuning_results_json', type=str)
    parser.add_argument('optimal_parameters_json', type=str)


    parser.add_argument('--cv_seed', type=int, default=29)
    parser.add_argument('--cv_n_folds', type=int, default=5)
    parser.add_argument('--opt_n_init', type=int, default=10)
    parser.add_argument('--opt_n_iter', type=int, default=50)
    parser.add_argument('--opt_random_seed', type=int, default=37)


    parser.add_argument('--learning_rate', nargs=2, type=float, default=(0.01, 0.3)) #metavar=('learning_rate_low', 'learning_rate_high')
    parser.add_argument('--gamma', nargs=2, type=float, default=(0.0, 0.3)) # metavar=('gamma_low', 'gamma_high')
    parser.add_argument('--min_child_weight', nargs=2, type=float,  default=(0.01, 0.07))
    parser.add_argument('--max_depth', nargs=2, type=int,  default=(3, 5)) # metavar=('max_depth_low', 'max_depth_high'),
    parser.add_argument('--subsample', nargs=2, type=float,  default=(0.7, 0.9)) # metavar=('subsample_low', 'subsample_high'),
    parser.add_argument('--reg_alpha', nargs=2, type=float,  default=(0.01, 0.1)) # metavar=('reg_alpha_low', 'reg_alpha_high'),
    parser.add_argument('--reg_lambda', nargs=2, type=float,  default=(0.01, 0.1)) # metavar=('reg_lambda_low', 'reg_lambda_high'),
    parser.add_argument('--col_sample_bytree', nargs=2, type=float,  default=(0.1, 0.5)) # metavar=('col_sample_bytree_low', 'col_sample_bytree_high'),

    
    # arff_filepath = './data/Pistachio_Dataset/Pistachio_16_Features_Dataset/Pistachio_16_Features_Dataset.arff'
    # parquet_path = './data/pistachio_16.snappy.pqt'
    args = parser.parse_args()

    # form dictionary defining boundaries of parameter space

    pbounds = {
        'learning_rate': (args.learning_rate[0], args.learning_rate[1]), 
        'gamma': (args.gamma[0], args.gamma[1]), 
        'min_child_weight': (args.min_child_weight[0], args.min_child_weight[1]), 
        'max_depth': (args.max_depth[0], args.max_depth[1]), 
        'subsample': (args.subsample[0], args.subsample[1]), 
        'reg_alpha': (args.reg_alpha[0], args.reg_alpha[1]), 
        'reg_lambda': (args.reg_lambda[0], args.reg_lambda[1]), 
        'colsample_bytree': (args.col_sample_bytree[0], args.col_sample_bytree[1])
    }

    model_tune(
        args.train_file,
        args.features_json,
        args.tuning_results_json,
        args.optimal_parameters_json,
        pbounds,
        cv_seed=args.cv_seed,
        n_folds=args.cv_n_folds,
        opt_n_init=args.opt_n_init,
        opt_n_iter=args.opt_n_iter,
        opt_random_seed=args.opt_random_seed
        )


if __name__ == "__main__":
    main()
