#!/bin/bash

#run pipeline components locally

TEST_DIR="$PWD/test_output"
DATA_DIR="$PWD/../data"

ARFF_FILE='/data/Pistachio_16_Features_Dataset.arff'
SCHEMA_FILE='/data/pistachio_schema.json'
PSI_FILE='/data/PSI_profile.pkl'
PSI_RESULTS_JSON='./test_output/test_psi_results.json'


IMAGE='pistachio_base:0.0.1'


TRAIN_PATH='/test_output/load_data_out/train.pqt'
TEST_PATH='/test_output/load_data_out/test.pqt'

PREPROC_TRAIN_PATH='/test_output/preproc_data_out/preproc_train.pqt'
PREPROC_TEST_PATH='/test_output/preproc_data_out/preproc_test.pqt'

if [ -d $TEST_DIR ]
then
  \rm -rf $TEST_DIR/*
fi

# lint image code
# install pylint in image and run it
docker run --rm -v $TEST_DIR:/test_output --entrypoint /bin/bash $IMAGE \ 
  "( pip install pylint && pylint . ) > /test_output/pylint_out.txt"

# run load_data

docker run --rm -v $TEST_DIR:/test_output -v $DATA_DIR:/data $IMAGE \
  "./load_data.py"  "$ARFF_FILE"  "$TRAIN_PATH"  "$TEST_PATH"  --split_seed 37  --test_fraction 0.2  --label_column Class

# validate data

docker run --rm -v $TEST_DIR:/test_output -v $DATA_DIR:/data $IMAGE \
  "./validate_data.py"  "$TRAIN_PATH"  "$SCHEMA_FILE"  

docker run --rm -v $TEST_DIR:/test_output -v $DATA_DIR:/data $IMAGE \
  "./validate_data.py"  "$TEST_PATH"  "$SCHEMA_FILE"  

# preprocess data

docker run --rm -v $TEST_DIR:/test_output -v $DATA_DIR:/data $IMAGE \
  "./preprocess_data.py"  "$TRAIN_PATH"  "$PREPROC_TRAIN_PATH"  

docker run --rm -v $TEST_DIR:/test_output -v $DATA_DIR:/data $IMAGE \
  "./preprocess_data.py"  "$TEST_PATH"  "$PREPROC_TEST_PATH"  

docker run --rm -v $TEST_DIR:/test_output -v $DATA_DIR:/data --entrypoint "./train_monitoring.py" $IMAGE \
  "$PREPROC_TRAIN_PATH"  "$PSI_FILE"  

docker run --rm -v $TEST_DIR:/test_output -v $DATA_DIR:/data --entrypoint "./infer_monitor.py" $IMAGE \
  "$PREPROC_TEST_PATH"  "$PSI_FILE" "$PSI_RESULTS_JSON"


