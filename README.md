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
