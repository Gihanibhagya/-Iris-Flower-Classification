from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Optional


try:
    model = joblib.load("model.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

app = FastAPI(
    title="Iris Flower Classification API",
    description="API for classifying iris flowers into setosa, versicolor, or virginica",
    version="1.0"
)


species_names = ["setosa", "versicolor", "virginica"]


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }


class PredictionOutput(BaseModel):
    species: str
    species_id: int
    probabilities: Optional[dict] = None
    confidence: Optional[float] = None

@app.get("/", tags=["Health Check"])
def health_check():
    return {"status": "healthy", "message": "Iris Classification API is running"}

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(input_data: IrisFeatures):
    try:
        # Convert input to numpy array
        features = np.array([
            [input_data.sepal_length, input_data.sepal_width, 
             input_data.petal_length, input_data.petal_width]
        ])

        
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        
        output = {
            "species": species_names[prediction],
            "species_id": int(prediction),
            "probabilities": {
                species_names[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            },
            "confidence": float(max(probabilities))
        }

        return output

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info", tags=["Model Information"])
def model_info():
    return {
        "model_type": "RandomForestClassifier",
        "problem_type": "classification",
        "classes": species_names,
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "feature_importance": dict(zip(
            ["sepal_length", "sepal_width", "petal_length", "petal_width"],
            [float(x) for x in model.feature_importances_]
        ))
    }