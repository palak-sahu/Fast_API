from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
import joblib
import numpy as np
import pandas as pd

# 1. Load the trained regression model
model = joblib.load('model/model.pkl')

# 2. Create FastAPI app instance
app = FastAPI(title="Medical Premium Prediction API")


# 3. Define the dropdown options for the UI
class BinaryChoice(str, Enum):
    yes = "Yes"
    no = "No"


# 4. Define the request body schema using Pydantic
class PredictRequest(BaseModel):
    Age: int
    Diabetes: BinaryChoice  # Changed to BinaryChoice for dropdown
    BloodPressureProblems: BinaryChoice
    AnyTransplants: BinaryChoice
    AnyChronicDiseases: BinaryChoice
    Height: int
    Weight: int
    KnownAllergies: BinaryChoice
    HistoryOfCancerInFamily: BinaryChoice
    NumberOfMajorSurgeries: int


# 5. Define the prediction endpoint
@app.post("/predict")
def predict(request: PredictRequest):
    # Convert "Yes"/"No" back to 1/0 for the model calculation
    # We create a dictionary and map the Enum values to integers
    input_data = request.dict()

    # Map the dropdown choices to 1 and 0
    mapping = {"Yes": 1, "No": 0}

    # Prepare the feature list in the exact order the model expects
    features = [
        input_data['Age'],
        mapping[input_data['Diabetes']],
        mapping[input_data['BloodPressureProblems']],
        mapping[input_data['AnyTransplants']],
        mapping[input_data['AnyChronicDiseases']],
        input_data['Height'],
        input_data['Weight'],
        mapping[input_data['KnownAllergies']],
        mapping[input_data['HistoryOfCancerInFamily']],
        input_data['NumberOfMajorSurgeries']
    ]

    # Convert to NumPy array for prediction
    data = np.array([features])

    # Make prediction
    prediction = model.predict(data)[0]

    # Return result
    return {
        "Predicted Premium Price": round(float(prediction), 2)
    }


# 6. Define a root endpoint for basic API check
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Medical Premium Prediction API",
        "instructions": "Go to /docs to use the interactive UI"
    }