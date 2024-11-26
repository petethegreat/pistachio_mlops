# pistachio-mlops

Came across a pistachio dataset for classification. Intent here is to use this to develop an ml-pipeline (kubeflow/vertex-ai) and deploy on gcp

will develop a simple model in a jupyterlab notebook, and use that as a starting point for pipeline development.




## Data
Pistachio Image Dataset downloaded from kaggle [here](https://www.kaggle.com/datasets/muratkokludataset/pistachio-image-dataset)

will use the 16 feature version which contains 1718 records across two pistachio types.

pandera for schema/data validation

## Notebook 

A jupyter notebook containing code used to train an XGBoost model for pistachio classification is [here](/notebook/pistachio.ipynb). The notebook loads, validates, and preprocesses the dataset, trains/tunes a model using Bayesian optimisation, and saves and evaluates the final model.



## Pipeline

The kubeflow pipeline definition for the model training is in [training_pipeline.py](/pipeline/pipeline_definition/training_pipeline.py).
The training pipeline consists of the following steps
  - loads and splits data (train/test)
  - validates train and test data against a defined schema
  - preprocesses train and test data
  - computes monitoring statistics (Population Stability Index) on training data, storing the results as an artifact
  - runs hyperparameter tuning using cross-validation/bayesian optimisation
  - tunes final model based on optimal parameters, storing the model in cloud storage and model registry
  - evaluates the final model on both the train and test datasets, sotring results (metrics/plots) in cloud storage and as KFP ClassificationMetrics

### Components
Components are defined in [components.py](/pipeline/pipeline_definition/components.py). Most of the core python code is stored in the docker image ("base_image" - see below), with the component definitions simply importing and invoking the relevant functions. Some components merely shuffle data/artifacts to/from GCP/Vertex AI services, these use the "gcp_aip_image".

### Images
  - [base image](/pipeline/images/base_image/Dockerfile) - Python image holding all the ML functionality
  - [gcp_aip_image](/pipeline/images/gcp_aip_image/Dockerfile) - python image with libraries for interacting with gcp services
  - [serving image](/pipeline/images/serving_image/Dockerfile) - python/FastApi image for serving model predictions. Loads model artifact from cloud storage.
 
Scripts:
  - build_images.sh - builds all of the above images (locally)
  - tag_and_push.sh - creates artifact registry tags for each of the local images, and pushes to artifact registry.
  - test_images.sh - Invokes all of the images - essentially running the training pipeline locally. KFP has a localRunner that does this, so should look into that.


# TODO

  - kfp has a local runner/docker setup for testing components. look at this instead of test_images.sh
  - XGboost warnings - can disable them in the container code - verbosity 0 or some other flag



  

