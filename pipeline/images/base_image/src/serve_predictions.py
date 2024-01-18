
from fastapi import FastAPI

from google.cloud import storage
from pydantic import BaseModel

import logging
import pandas as pd

logger = logging.getLogger('pistachio')
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

MODEL = None

class InstanceModel(BaseModel):
    AREA: float
    PERIMETER:  float
    MAJOR_AXIS:  float
    MINOR_AXIS:  float
    ECCENTRICITY:  float
    EQDIASQ:  float
    SOLIDITY:  float
    CONVEX_AREA:  float
    EXTENT:  float
    ASPECT_RATIO:  float
    ROUNDNESS:  float
    COMPACTNESS:  float
    SHAPEFACTOR_1:  float
    SHAPEFACTOR_2:  float
    SHAPEFACTOR_3:  float
    SHAPEFACTOR_4:  float
    SOLIDITY_MAJOR: float


class InferenceBody(BaseModel):
    instances: List[InferenceRecord]


def load_model_from_gcs(path: str):
    """load model from gcs"""
    logger.info(f'loading model from gcs: {path}')

    # path starts with gs://
    bucket_index = path.index('/',5)
    bucket_name = path[5:bucket_index]
    blob_path = path[bucket_index+1:] + "model_pickle.pkl"

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_path)
    bstr = blob.dowload_as_bytes()
    model = pickle.loads(bstr)
    logger.info('model loaded from gcs')
    return model
     



def load_model_from_local_storage(path: str):
    """load model from storage"""

    logger.info('loading model from local file')

    model_path = os.path.join(path, 'model_pickle.pkl')
    with open(model_path, 'rb') as infile:
        model = pickle.loads(infile)
    logger.info(f'loaded model from {path}')
    
    return model


def get_model_artifacts(path: str):
    """get artifacts from either local storage (for testing) or gcs"""
    if path.startswith('gs://'):
        model = load_model_from_gcs(path)
        return model 
    # else load from local storage
    return load_model_from_local_storage(path)


app = FastAPI()

model_path = os.env['AIP_STORAGE_URI']

MODEL = get_model_artifacts(model_path)



# post request body:
# {
#   "instances": INSTANCES,
#   "parameters": PARAMETERS
# }


# https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#prediction

@app.post("/predict")
async def predict(inference_data: InferenceBody):
    """handles the request, return prediction"""

    # get data in df
    inference_data = pd.DataFrame.from_records([dict[x] for x in inference_data.instances])
    logger.info(inference_data.head())
    features = inference_data.columns

    inference_data['inferred_class'] = MODEL.predict(inference_data[features])
    inference_data['class_1_probability'] = MODEL.predict_proba(inference_data[features])[:,1]

    # output
    output_df = inference_data[['inferred_class', 'class_1_probability']]

    output = { "predictions": output_df.to_dict('records')}
    return output


@app.get("/health")
async def health():
    """handles the request, return prediction"""   
    # either respond 200 if ready to receive prediction requests, or 503 if not.
    return 200 if MODEL is not None else 503

    

