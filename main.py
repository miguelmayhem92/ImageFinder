from fastapi import FastAPI, BackgroundTasks, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, PlainTextResponse
import schemas
from pydantic import BaseModel

app = FastAPI()

@app.get("/health")
def health():
    health = schemas.Health()
    return health.dict()

"""@app.post("/predict_with_production_model")
def launch_ImageFinder(image):

    return {'gato'}"""