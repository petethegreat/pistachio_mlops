from python:3.11-slim-bullseye

copy requirements.txt /requirements.txt
run pip3 install -r /requirements.txt --no-cache-dir
copy ./src/ /base_image
workdir /base_image

entrypoint ["python"]

