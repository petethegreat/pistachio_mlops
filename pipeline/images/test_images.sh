#!/bin/bash



#run pipeline components locally

TEST_DIR="$PWD/test_output"
DATA_DIR="$PWD/../data"

ARFF_FILE='/data/Pistachio_16_Features_Dataset.arff'
SCHEMA_FILE='/data/pistachio_schema.json'
PSI_FILE='/data/PSI_profile.pkl'
PSI_RESULTS_JSON='./test_output/test_psi_results.json'


IMAGE='pistachio_base:0.0.1'

# these are all paths *in* the container, '/test_output/' should be specified absolutely
TRAIN_PATH="/test_output/load_data_out/train.pqt"
TEST_PATH="/test_output/load_data_out/test.pqt"

PREPROC_TRAIN_PATH="/test_output/preproc_data_out/preproc_train.pqt"
PREPROC_TEST_PATH="/test_output/preproc_data_out/preproc_test.pqt"
FEATURE_LIST_PATH="/test_output/preproc_data_out/features.json"

OPTIMAL_PARAMETERS_PATH="/test_output/tuning/optimal_parameters.json"
TUNING_RESULTS_PATH="/test_output/tuning/tuning_details.json"

MODEL_ARTIFACT_PATH="/test_output/model_training/model.pkl"

EVALUATE_TRAIN_FI_PLOT_PATH="/test_output/train_evaluation/feature_importance.png"
EVALUATE_TRAIN_METRICS_PATH="/test_output/train_evaluation/metrics.json"

EVALUATE_TEST_FI_PLOT_PATH="/test_output/test_evaluation/feature_importance.png"
EVALUATE_TEST_METRICS_PATH="/test_output/test_evaluation/metrics.json"

if [ -d $TEST_DIR ]
then
  echo "deleting $TEST_DIR"
  docker run --rm -v $TEST_DIR:/test_output --entrypoint /bin/bash $IMAGE  -c "rm -r  /test_output/*"
fi

# lint image code
# install pylint in image and run it
echo "linting code"
docker run --rm -v $TEST_DIR:/test_output --entrypoint /bin/bash $IMAGE   -c " (pylint pistachio) > /test_output/pylint_out.txt"
echo "pylint status: $?"
# run load_data

# stop execution on error
set -e

echo "running load_data"
docker run --rm -v $TEST_DIR:/test_output -v $DATA_DIR:/data $IMAGE \
  "./load_data.py"  "$ARFF_FILE"  "$TRAIN_PATH"  "$TEST_PATH"  --split_seed 37  --test_fraction 0.2  --label_column Class

# validate data
echo "running validate_data (train)"
docker run --rm -v $TEST_DIR:/test_output -v $DATA_DIR:/data $IMAGE \
  "./validate_data.py"  "$TRAIN_PATH"  "$SCHEMA_FILE"  

echo "running validate_data (test)"

docker run --rm -v $TEST_DIR:/test_output -v $DATA_DIR:/data $IMAGE \
  "./validate_data.py"  "$TEST_PATH"  "$SCHEMA_FILE"  

# preprocess data
echo "running preprocess data (train)"
docker run --rm -v $TEST_DIR:/test_output -v $DATA_DIR:/data $IMAGE \
  "./preprocess_data.py"  "$TRAIN_PATH"  "$PREPROC_TRAIN_PATH" "$FEATURE_LIST_PATH" 

echo "running preprocess data (test)"
docker run --rm -v $TEST_DIR:/test_output -v $DATA_DIR:/data $IMAGE \
  "./preprocess_data.py"  "$TEST_PATH"  "$PREPROC_TEST_PATH" "/test_output/junk.json" 

docker run --rm -v $TEST_DIR:/test_output -v $DATA_DIR:/data --entrypoint "./train_monitoring.py" $IMAGE \
  "$PREPROC_TRAIN_PATH"  "$PSI_FILE"  

docker run --rm -v $TEST_DIR:/test_output -v $DATA_DIR:/data --entrypoint "./infer_monitor.py" $IMAGE \
  "$PREPROC_TEST_PATH"  "$PSI_FILE" "$PSI_RESULTS_JSON"
echo "running model tuning"
docker run --rm -v $TEST_DIR:/test_output -v $DATA_DIR:/data --entrypoint "./model_tuning.py" $IMAGE \
  "$PREPROC_TRAIN_PATH"  "$FEATURE_LIST_PATH" "$TUNING_RESULTS_PATH" "$OPTIMAL_PARAMETERS_PATH" "--cv_seed" "37"

echo "running model training"
docker run --rm -v $TEST_DIR:/test_output -v $DATA_DIR:/data --entrypoint "./train_model.py" $IMAGE \
  "$PREPROC_TRAIN_PATH"  "$OPTIMAL_PARAMETERS_PATH" "$MODEL_ARTIFACT_PATH" "$FEATURE_LIST_PATH" 

echo "running model evaluation on train data"
docker run --rm -v $TEST_DIR:/test_output -v $DATA_DIR:/data --entrypoint "./evaluate_model.py" $IMAGE \
  "$PREPROC_TRAIN_PATH"  "$MODEL_ARTIFACT_PATH" "$FEATURE_LIST_PATH" "$EVALUATE_TRAIN_METRICS_PATH" "$EVALUATE_TRAIN_FI_PLOT_PATH" "--metric_prefix" "train_metrics_"

echo "running model evaluation on test data"
docker run --rm -v $TEST_DIR:/test_output -v $DATA_DIR:/data --entrypoint "./evaluate_model.py" $IMAGE \
  "$PREPROC_TEST_PATH"  "$MODEL_ARTIFACT_PATH" "$FEATURE_LIST_PATH" "$EVALUATE_TEST_METRICS_PATH" "$EVALUATE_TEST_FI_PLOT_PATH" "--metric_prefix" "test_metrics_"


echo "running model evaluation on test data"
docker run --rm -v $TEST_DIR:/test_output -v $DATA_DIR:/data --entrypoint /bin/bash $IMAGE 