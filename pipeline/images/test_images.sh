#!/bin/bash

#run pipeline components locally

TEST_DIR="$PWD/test_output"
DATA_DIR="$PWD/../data"

ARFF_FILE='/data/Pistachio_16_Features_Dataset.arff'
SCHEMA_FILE='/data/pistachio_schema.json'

IMAGE='pistachio_base:0.0.1'

TRAIN_PATH='/test_output/train.pqt'
TEST_PATH='/test_output/test.pqt'

# run load_data

docker run --rm -v $TEST_DIR:/test_output -v $DATA_DIR:/data $IMAGE \
  "./load_data.py"  "$ARFF_FILE"  "$TRAIN_PATH"  "$TEST_PATH"  --split_seed 37  --test_fraction 0.2  --label_column Class

docker run --rm -v $TEST_DIR:/test_output -v $DATA_DIR:/data $IMAGE \
  "./validate_data.py"  "$TRAIN_PATH"  "$SCHEMA_FILE"  

docker run --rm -v $TEST_DIR:/test_output -v $DATA_DIR:/data $IMAGE \
  "./validate_data.py"  "$TEST_PATH"  "$SCHEMA_FILE"  
