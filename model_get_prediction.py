from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris

app = FastAPI()

# Load Iris dataset to get target names
iris = load_iris()
target_names = iris.target_names  # Example: ['setosa', 'versicolor', 'virginica']

def load_model(model_name: str, version_number: str):
    client = mlflow.tracking.MlflowClient()
    model_uri = f"models:/{model_name}/{version_number}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model version {version_number} not found: {e}")

class InputData(BaseModel):
    features: list[float]

@app.post("/predict/{version_number}")
async def predict(version_number: str, input_data: InputData):
    model_name = "DecisionTreeClassifier"
    model = load_model(model_name, version_number)
    try:
        features = np.array(input_data.features).reshape(1, -1)
        prediction_index = model.predict(features)[0]
        prediction_label = target_names[prediction_index]  # Convert index to label
        return {"prediction": prediction_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
