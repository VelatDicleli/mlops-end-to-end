from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


prediction_pipeline = PredictPipeline(
    preprocessor_path="artifacts/preprocessor.pkl",
    model_path="artifacts/model.pkl"
)


@app.post("/predict")
def predict(request: CustomData):
    

    
    features = request.get_features()
    prediction = prediction_pipeline.predict(features)
    
    return {"prediction": prediction.tolist()}