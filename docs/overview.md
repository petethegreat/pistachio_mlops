# Overview

Intent of this project is to set up Machine Learning pipelines related to training and scoring a classification model.
Initial development of the model is carried out in a notebook. The keys steps worked through in model development (data processing, model tuning/training, performance evaluation, etc) are then used to construct a kubeflow pipeline for model training. This is being run as a vertex ai pipeline in gcp, and makes use of several google services. A pipeline for batch inference is also created.

## Design
The objective was to seperate the "core" data analysis/modelling/evaluation code from the kfp pipeline-specific elements. Common functionality for these tasks can be used across many projects and teams within an org, and may be wrapped into a python package that can be pip installed into a notebook environment. In this project, this code is contained within the image definition (`pipeline/images/base_image/src/pistachio` in this repo), but in principle the image could install a particular version of this as a package.

## pipelines
Pipeline definitions are located in `pipeline/pipeline_definition` - `training_pipeline.py` for training and `batch_prediction_pipeline.py` for prediction. Components for both are in `components.py` - in most cases the components are thin wrappers around functions from the pistachio module loaded into the base_image. Some things are a little more involved when more juggling of kfp paths or artifacts are required, or when using google cloud pipeline components/AI platform functions.





