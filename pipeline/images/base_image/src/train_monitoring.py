#!/usr/bin/env python
"""train_monitoring
generate data data profile at train time, to be used when monitering data during inference

"""

from argparse import ArgumentParser
from pistachio.data_handling import load_parquet_file
from pistachio.psi_metrics import PSImetrics
from pistachio.utils import ensure_directory_exists

import os

import logging
import sys
## logging

logger = logging.getLogger('pistachio')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def fit_psi(train_data_path: str, psi_artifact_path: str) -> None:
    """fit and save PSI object"""
    logger.info('initialising PSImetrics')
    psi = PSImetrics()

    psi.continuous_cols = ['AREA', 'PERIMETER', 'MAJOR_AXIS', 'MINOR_AXIS', 'ECCENTRICITY',
       'EQDIASQ', 'SOLIDITY', 'CONVEX_AREA', 'EXTENT', 'ASPECT_RATIO',
       'ROUNDNESS', 'COMPACTNESS', 'SHAPEFACTOR_1', 'SHAPEFACTOR_2',
       'SHAPEFACTOR_3', 'SHAPEFACTOR_4', 'SOLIDITY_MAJOR']
    psi.categorical_cols = ['Class', 'Target']
    logger.info('loading train data ')

    train_data = load_parquet_file(train_data_path)
    logger.info('fitting train data ')

    psi.fit(train_data)

    ensure_directory_exists(psi_artifact_path)


    psi.save(psi_artifact_path)
    logger.info(f'saved psi object to {psi_artifact_path}')
##############################################################################

def main():
    """do the things"""
    parser = ArgumentParser(
        description="record data profile at training time so that population stability index can be compited during inference"
    )
    parser.add_argument('train_data_path', type=str)
    parser.add_argument('psi_artifact_path', type=str)
    args = parser.parse_args()
    fit_psi(args.train_data_path, args.psi_artifact_path)

if __name__ == '__main__':
    main()


