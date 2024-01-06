#!/bin/bash

docker build -t pistachio_base:0.0.1 ./base_image
docker build -t pistachio_gcp_aip:0.0.1 ./gcp_aip_image


