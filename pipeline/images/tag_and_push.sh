#!/bin/bash

# tag and push

docker image tag pistachio_base:0.0.1 northamerica-northeast2-docker.pkg.dev/pistachio-mlops-sbx/pistachio-base/pistachio_base:0.0.1
docker image push northamerica-northeast2-docker.pkg.dev/pistachio-mlops-sbx/pistachio-base/pistachio_base:0.0.1

docker image tag pistachio_gcp_aip:0.0.1 northamerica-northeast2-docker.pkg.dev/pistachio-mlops-sbx/pistachio-base/pistachio_gcp_aip:0.0.1
docker image push northamerica-northeast2-docker.pkg.dev/pistachio-mlops-sbx/pistachio-base/pistachio_gcp_aip:0.0.1
