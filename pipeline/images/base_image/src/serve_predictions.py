
from fastapi import FastAPI

from google.cloud import storage


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
async def predict(instances, parameters):
    """handles the request, return prediction"""


@app.get("/health")
async def predict(instances, parameters):
    """handles the request, return prediction"""   
    # either respond 200 if ready to receive prediction requests, or 503 if not.

