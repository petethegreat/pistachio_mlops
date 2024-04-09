
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Request

from google.cloud import storage
from pydantic import BaseModel

import logging
import pandas as pd
import numpy as np

import sys 
import os

import pickle
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger('pistachio')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

logger.info('serving container starting')

# this needs to be a mutable type, else assignments in functions are stuck in that scope
MODEL_ARTIFACTS = {}

class InferenceRequestModel(BaseModel):
    instances: List[List[float]]
    parameters: Optional[Dict[str,str]] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "instances": [
                        [ 77937.0,1291.8571,432.2187,263.1985,0.7932,315.0119,0.9005,86550.0,0.6985,1.6422,0.5868,0.7288,0.0055,0.0034,0.5312,0.8723,389.21293935],
                        [ 84839.0,2476.2419,464.7022,251.1368,0.8414,328.6645,0.9127,92958.0,0.7131,1.8504,0.1739,0.7073,0.0055,0.003,0.5002,0.9256,424.13369794]
                    ],
                    "parameters": {
                        "key": "value",
                        "cat": "dog"
                    }
                }
            ]
        }
    }

# logger.info('instance model defined')


def load_model_from_gcs(path: str, local_path: str='./model_pickle.pkl'):
    """load model from gcs"""
    logger.info(f'loading model from gcs: {path}')

    client = storage.Client()
    # the_path = urllib.parse.urljoin(path,"model_pickle.pkl")
    the_path = os.path.join(path,"model_pickle.pkl")
    logger.info(f'the_path = {the_path}')


    with open(local_path, 'wb') as model_file:
        client.download_blob_to_file(the_path, model_file)
    
    model = load_model_from_local_storage(local_path)
    return model

    



    # path starts with gs://
    # bucket_index = path.index('/',5)
    # bucket_name = path[5:bucket_index]
    # blob_path = path[bucket_index+1:] + "model_pickle.pkl"

    # bucket = client.get_bucket(bucket_name)
    # blob = bucket.blob(blob_path)
    # bstr = blob.download_as_bytes()
    # model = pickle.loads(bstr)
    # logger.info('model downloaded from gcs')
    # return model
     



def load_model_from_local_storage(path: str):
    """load model from storage"""

    logger.info('loading model from local file')
    if not path.endswith('.pkl'):
        model_path = os.path.join(path, 'model_pickle.pkl')
    else:
        model_path = path
    with open(model_path, 'rb') as infile:
        model = pickle.load(infile)
    logger.info(f'loaded model from {path}')
    logger.info(model)
    
    return model


def get_model_artifacts(path: str):
    """get artifacts from either local storage (for testing) or gcs"""
    if path.startswith('gs://'):
        model =  load_model_from_gcs(path)
    else:
        model =  load_model_from_local_storage(path)
    # else load from local storage
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    model_path = os.environ['AIP_STORAGE_URI']
    MODEL_ARTIFACTS['model'] = get_model_artifacts(model_path)
    logger.info(f'starting up, model = {MODEL_ARTIFACTS["model"]}')
    yield
    # Clean up the ML models and release the resources
    MODEL_ARTIFACTS.clear()


app = FastAPI(lifespan=lifespan)

# startup envent - load model
# DEPRECATED
# @app.on_event("startup")
# async def startup_event():
#     model_path = os.environ['AIP_STORAGE_URI']
#     logger.info(f'getting model from {model_path}')
#     MODEL = await get_model_artifacts(model_path)
#     logger.info(f"model: {MODEL}")



# post request body:
# {
#   "instances": INSTANCES,
#   "parameters": PARAMETERS
# }

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )



# https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#prediction

@app.post("/predict")
async def predict(infer_request: InferenceRequestModel):
    """handles the request, return prediction"""

    logger.warn(f'received request')
    # return await request.json()
    
    # data = await request.json()
    instances = infer_request.instances

    # logger.warn(data)
    columns = [
        "AREA",
        "PERIMETER",
        "MAJOR_AXIS",
        "MINOR_AXIS",
        "ECCENTRICITY",
        "EQDIASQ",
        "SOLIDITY",
        "CONVEX_AREA",
        "EXTENT",
        "ASPECT_RATIO",
        "ROUNDNESS",
        "COMPACTNESS",
        "SHAPEFACTOR_1",
        "SHAPEFACTOR_2",
        "SHAPEFACTOR_3",
        "SHAPEFACTOR_4",
        "SOLIDITY_MAJOR"
    ]

    # instances = data['instances']
    logger.info(instances)

    predictions = None
    inference_df = pd.DataFrame(instances, columns=columns)
    model = MODEL_ARTIFACTS['model']
    inferred_class = model.predict(inference_df)
    class_1_probability = model.predict_proba(inference_df)[:,1]

    output_df = pd.DataFrame({'inferred_class':inferred_class, 'class_1_probability': class_1_probability})
    predictions = output_df.to_dict('records')
    output = { "predictions": predictions}
    return output


    # try:
        
    #     logger.info(f'instances: {instances}')
    #     inference_data_array = np.asarray(instances)


    #     # get data in df
    #     # inference_data = pd.DataFrame.from_records([dict(x) for x in inference_data.instances])
    #     inference_data_array = np.asarray(inference_data.instances)
    #     logger.info(inference_data_array[0:2,:])
    #     # features = inference_data.columns
    # except Exception as e:
    #     output = { "predictions": [{'error': 'could not gather features for inference'}, {'instances':instances}]}


    # try:
    #     model = MODEL_ARTIFACTS['model']
    #     # inference_data['inferred_class'] = model.predict(inference_data[features])
    #     # inference_data['class_1_probability'] = model.predict_proba(inference_data[features])[:,1]
    #     inferred_class = model.predict(inference_data_array)
    #     class_1_probability = model.predict_proba(inference_data_array)[:,1]
    #     output_df = pd.DataFrame({'inferred_class':inferred_class, 'class_1_probability': class_1_probability})
    #     predictions = output_df.to_dict('records')

    # except Exception as e:
    #     predictions = [{'error': 'could not infer from model'} for x in range(inference_data_array.shape[0]) ]

    # # output

    # output = { "predictions": predictions}
    # return output


@app.get("/health")
async def health():
    """handles the request, return prediction"""   
    # either respond 200 if ready to receive prediction requests, or 503 if not.
    logger.info(MODEL_ARTIFACTS.get('model'))
    if 'model' not in MODEL_ARTIFACTS:
        raise HTTPException(status_code=503, detail=f"Model not loaded")
    return {}

    

