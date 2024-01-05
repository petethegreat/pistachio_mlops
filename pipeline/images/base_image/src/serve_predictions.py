
from fastapi import FastAPI

from google.cloud import storage
from pydantic import BaseModel

class InferenceRecord(BaseModel):
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

class InferenceBody(BaseModel):
    instances: List[InferenceRecord]



app = FastAPI()

def get_model_artifacts(path: str):
    """get artifacts from either local storage (for testing) or gcs"""


# post request body:
# {
#   "instances": INSTANCES,
#   "parameters": PARAMETERS
# }


# https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#prediction

@app.post("/predict")
async def predict(instances):
    """handles the request, return prediction"""


@app.get("/health")
async def predict(InferenceBody):
    """handles the request, return prediction"""   
    # either respond 200 if ready to receive prediction requests, or 503 if not.
    

