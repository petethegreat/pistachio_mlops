# pistachio-mlops

Came across a pistachio dataset for classification. Intent here is to use this to develop an ml-pipeline (kubeflow/vertex-ai) and deploy on gcp

will develop a simple model in a jupyterlab notebook, and use that as a starting point for pipeline development.



## Data
Pistachio Image Dataset downloaded from kaggle [here](https://www.kaggle.com/datasets/muratkokludataset/pistachio-image-dataset)

will use the 16 feature version which contains 1718 records across two pistachio types.

pandera for schema/data validation

installing packages into image:
[just use pip](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/recipes.html#using-mamba-install-recommended-or-pip-install-in-a-child-docker-image)


## Pipeline
  - base image for all python functionality
  - python scripts to handle arguments for each component definition
  
  things:
    - some sort of config, project, storage, arifact locations, etc.
    - build images locally and push to AR, vs cloudbuild
    - component definitions, including image location

https://www.kubeflow.org/docs/components/pipelines/v1/sdk/component-development/#creating-a-component-specification

## images

test load_data
```docker run  -v ./pipeline/data:/data pistachio_base:0.0.1 load_data.py /data/Pistachio_16_Features_Dataset.arff /data/pistachio_imagetest_train.pqt /data/pistachio_imagetest_test.pqt```

test validate_data
```docker run  -v ./pipeline/data:/data pistachio_base:0.0.1 validate_data.py /data/pistachio_imagetest_train.pqt /data/pistachio_schema.json```



<!-- https://stackoverflow.com/questions/68348026/run-id-in-kubeflow-pipelines-on-vertex-ai
dsl.PIPELINE_JOB_ID_PLACEHOLDER

https://github.com/GoogleCloudPlatform/professional-services/blob/main/examples/vertex_pipeline/components/component_base/src/train.py -->

# TODO

  - kfp has a local runner/docker setup for testing components. look at this instead of test_images.sh
  - XGboost warnings - can disable them in the container code - verbosity 0 or some other flag



  

