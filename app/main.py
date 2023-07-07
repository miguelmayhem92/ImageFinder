from fastapi import FastAPI, UploadFile, File

from utils import prediction
from schemas import Health

import os
import shutil

tmp_image_dir = 'tmp_image/'
project_name = 'ImageFinder'
app_version = "0.0.1"
model_version = "1"

app = FastAPI()

@app.get("/health")
def health():
    """
    Root Get
    """
    health = Health(
        name=project_name, api_version=app_version, model_version=model_version
    )

    return health.dict()

@app.post("/image_finder")
async def launch_ImageFinder(file: UploadFile = File(...)):

    file.filename = 'new_image.jpg'
    contents = await file.read()

    if not os.path.exists(tmp_image_dir):
        os.makedirs(tmp_image_dir)
    
    with open(f'{tmp_image_dir}{file.filename}', 'wb') as f:
        f.write(contents)

    result = prediction()

    shutil.rmtree(tmp_image_dir)

    return result


##################################################
#### testing mlflow simple artifacts in docker
#############################################

from pydantic import BaseModel
from typing import Text
import mlflow

path = os.getcwd()

# mlruns_path =  "http://localhost:5000"
# mlflow.set_tracking_uri(mlruns_path) 

class endpoint_input(BaseModel):
    content :Text = 'Gato'


def save_artifact(text):
    
    my_dict = {'text':text}
    with mlflow.start_run() as run:
        mlflow.log_dict(my_dict,"settings.json")
    return 'save in mlflow !!'

@app.post("/input") 
def save_dict(endpoint_input_:endpoint_input):
    result = save_artifact(endpoint_input_.content)
    return {result}

class endpoint_output(BaseModel):
    content :Text = 'asdwq25154qewq2121'

def call_artifact(runid):
        
    ticket_settings = mlflow.artifacts.load_dict(
        f'runs:/{runid}/settings.json'
    )
    return ticket_settings 

@app.post("/output")
def get_settings_endpoint(stock_code_:endpoint_output):

    settings_object = call_artifact(stock_code_.content)

    return settings_object