#! /usr/bin/env python
# preprocess data - cleaning/feature engineering. load into format for model training/consumption

from argparse import ArgumentParser
from pistachio.data_handling import load_parquet_file, preprocess_data


import logging
import sys
## logging

logger = logging.getLogger('pistachio')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
