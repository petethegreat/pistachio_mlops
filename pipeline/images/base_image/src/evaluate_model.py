#!/usr/bin/env python
"""evaluate_model.py

take a dataset and model artifact, generate metrics and plots
"""



from argparse import ArgumentParser
from pistachio.data_handling import load_parquet_file, validate_data_with_schema

import logging
import sys

## logging
logger = logging.getLogger('pistachio')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

#########################################################

def evaluate_model():


#########################################################
def main():

    parser = ArgumentParser(
        description="evaluate model on a dataset and generate performance metrics and plots"
    )
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('model_pickle_path', type=str)
    parser.add_argument('featurelist_json', type=str)
    
    parser.add_argument('--metric_prefix', type=str)
    parser.add_argument('--plot_title', type=str)
    parser.add_argument('--feature_importance_plot_path', type=str)
    parser.add_argument('featurelist_json', type=str)
    parser.add_argument('featurelist_json', type=str)
    parser.add_argument('featurelist_json', type=str)




    args = parser.parse_args()

    evaluate_model(

        )

if __name__ == "__main__":
    main()