from fastapi import FastAPI, UploadFile, File
from jobs.execute_ImageFinder import prediction
import schemas

import os
import shutil

tmp_image_dir = 'tmp_image/'

app = FastAPI()

@app.get("/health")
def health():
    health = schemas.Health()
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