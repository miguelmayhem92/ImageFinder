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