import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# Initialize the FastAPI app
app = FastAPI()

# Read the model name from environment variable or default to "Student_performance"
MODEL_NAME = os.getenv("MODEL_NAME", "Student_performance")
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    # Add other features as required

class PredictionResponse(BaseModel):
    prediction: float

def get_production_model_version(name):
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    latest_versions = client.search_model_versions(f"name='{name}'")
    for version in latest_versions:
        if version.current_stage == "Production":
            return version.version
    raise ValueError(f"No production model found for {name}")

def load_production_model(name):
    try:
        version = get_production_model_version(name)
        model_uri = f"models:/{name}/{version}"
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load production model: {str(e)}")

# Load the production model on startup
model = load_production_model(MODEL_NAME)

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        features = [request.feature1, request.feature2]  # Add other features as required
        prediction = model.predict([features])[0]
        return PredictionResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the student performance prediction API!"}

@app.get("/models")
def list_models():
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    return [{"version": version.version, "stage": version.current_stage} for version in versions]
