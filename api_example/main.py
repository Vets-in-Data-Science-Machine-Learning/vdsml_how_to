from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
from sklearn.datasets import load_iris

# Load the iris dataset and the trained model
iris = load_iris()
model = joblib.load("iris_model.pkl")

app = FastAPI()

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisPrediction(BaseModel):
    species: str

@app.post("/predict", response_model=IrisPrediction)
def predict_species(features: IrisFeatures):
    data = np.array([[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]])
    prediction = model.predict(data)
    species = iris.target_names[prediction[0]]
    return IrisPrediction(species=species)
