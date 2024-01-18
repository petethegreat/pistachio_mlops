# pistachio-mlops

Came across a pistachio dataset for classification. Intent here is to use this to develop an ml-pipeline (kubeflow/vertex-ai) and deploy on gcp

will develop a simple model in a jupyterlab notebook, and use that as a starting point for pipeline development.

## Data
Pistachio Image Dataset downloaded from kaggle [here](https://www.kaggle.com/datasets/muratkokludataset/pistachio-image-dataset)

will use the 16 feature version which contains 1718 records across two pistachio types.
## Notebook

build docker image, from notebook/docker dirctory
```bash
docker build -t pistachio_notebook:latest .
```

```bash
docker run --rm  --name jupy -p 6372:8888 -v ${PWD}/notebook:/home/jovyan/work/pistachio pistachio_notebook:latest
```

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

## Compoents

components dir is redundant, container specs can be defined in python in kfp v2


# todo
 - config - project id, artifact registry locations, etc.
 - build image
 - generate component definitions. (script, load stuff from config.)
 - build pipeline, push pipeline, run it.


<!-- https://stackoverflow.com/questions/68348026/run-id-in-kubeflow-pipelines-on-vertex-ai
dsl.PIPELINE_JOB_ID_PLACEHOLDER

https://github.com/GoogleCloudPlatform/professional-services/blob/main/examples/vertex_pipeline/components/component_base/src/train.py -->

# TODO
 - modify metadata of datasets to contain list of features/targets, don't need to pass this around as an artifact.
  - could hold a pointer to PSI this way also?
 - modify paths of png files output_plot.path = output_plot.path + '.png', same with json

- move components from container components to lightweight components - can specify our image as base image. still leaves image definition seperate from pipeline defnition.


## Batch prediction
  - current code does not do the preprocessing. Can get this to use a fastapi image
  - need to pass `--port $AIP_HTTP_PORT` in the args to the component, so that fastapi gets the port it should use.

  

