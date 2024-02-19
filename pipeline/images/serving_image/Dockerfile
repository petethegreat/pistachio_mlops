from python:3.11-slim-bullseye

copy requirements.txt /requirements.txt
run pip3 install -r /requirements.txt --no-cache-dir
copy ./src/ /serving_image
workdir /serving_image

EXPOSE 8080

entrypoint ["uvicorn"]
CMD ["serve_predictions:app", "--port", "8080", "--host", "0.0.0.0"]

