
## Notebook

build docker image, from notebook/docker dirctory
```bash
docker build -t pistachio_notebook:latest .
```

```bash
docker run --rm  --name jupy -p 6372:8888 -v ${PWD}/notebook:/home/jovyan/work/pistachio pistachio_notebook:latest
```

This assumes that the data file (not tracked by this repo) is present at `notebook/data/Pistachio_Dataset/Pistachio_16_Features_Dataset/Pistachio_16_Features_Dataset.arff`. It can be obtained from [here](https://www.kaggle.com/datasets/muratkokludataset/pistachio-image-dataset)


