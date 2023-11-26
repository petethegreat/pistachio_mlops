#!/usr/bin/env python
""" infer_monitor.py
run data monitoring at inference time
"""

from argparse import ArgumentParser
from pistachio.data_handling import load_parquet_file
from pistachio.psi_metrics import PSImetrics
import os

import logging
import sys
import json
## logging

logger = logging.getLogger('pistachio')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def eval_psi(dataset_path: str, psi_artifact_path: str, psi_results_path: str) -> None:
    """load PSI object and evaluate data

    Args:
        dataset_path (str): _description_
        psi_artifact_path (str): path to saved (fit) PSIMetrics object
    """
    logger.info('loading PSImetrics')
    psi = PSImetrics.load(psi_artifact_path)

    data = load_parquet_file(dataset_path)
    logger.info(f'evaluating data at {dataset_path} ')

    psi_values, details = psi.evaluate(data)

    #TODO - handle psi_values as metadata
    # - do this in pipeline code. have a small component to read the detail json and create artifacts 
    # of typekfp.dsl.Metric  and/or kfp.dsl.MarkDown
    
    # just log the psi_values here
    logger.info('PSI Values:')
    for pv in psi_values:
        logger.info(pv)
    

    output_dir = os.path.dirname(psi_results_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(psi_results_path,'w') as outfile:
        json.dump(details, outfile, indent=4)

    logger.info(f'saved psi results to {psi_results_path}')



def main():
    """do the things"""
    parser = ArgumentParser(
        description="record data profile at training time so that population stability index can be compited during inference"
    )
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('psi_artifact_path', type=str)
    parser.add_argument('psi_results_path', type=str)
    args = parser.parse_args()
    eval_psi(args.dataset_path, args.psi_artifact_path, args.psi_results_path)

if __name__ == '__main__':
    main()


