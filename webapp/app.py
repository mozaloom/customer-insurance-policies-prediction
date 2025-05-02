from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
from typing import Any

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
BAGGING_MODEL_PATH = os.path.join(MODEL_DIR, "bagging_model.pkl")
BOOSTING_MODEL_PATH = os.path.join(MODEL_DIR, "boosting_model.pkl")

FEATURE_COLUMNS = [
    "Age", "Driving_License", "Region_Code", "Previously_Insured",
    "Annual_Premium", "Policy_Sales_Channel", "Vintage",
    "Gender_Female", "Gender_Male",
    "Vehicle_Age_< 1 Year", "Vehicle_Age_1-2 Year", "Vehicle_Age_> 2 Years",
    "Vehicle_Damage_No", "Vehicle_Damage_Yes"
]

# --- Data Models ---
class PolicyRequest(BaseModel):
    age: int = Field(..., ge=0)
    driving_license: int = Field(..., ge=0, le=1)
    region_code: int
    previously_insured: int = Field(..., ge=0, le=1)
    annual_premium: float = Field(..., ge=0)
    policy_sales_channel: int
    vintage: int = Field(..., ge=0)
    gender: str = Field(..., pattern="^(Male|Female)$")
    vehicle_age: str = Field(..., pattern="^(< 1 Year|1-2 Year|> 2 Years)$")
    vehicle_damage: str = Field(..., pattern="^(Yes|No)$")
    model_type: str = Field("bagging", pattern="^(bagging|boosting)$")

class PredictionResponse(BaseModel):
    model: str
    prediction: int
    probability: float

# --- Utility Functions ---
def load_model(path: str) -> Any:
    """Load a model from disk, raising HTTPException if it fails."""
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model at {path}: {e}")

def make_feature_vector(req: PolicyRequest) -> np.ndarray:
    """Convert request to feature vector as expected by the model."""
    fv = dict.fromkeys(FEATURE_COLUMNS, 0)
    # Numeric features
    fv["Age"] = req.age
    fv["Driving_License"] = req.driving_license
    fv["Region_Code"] = req.region_code
    fv["Previously_Insured"] = req.previously_insured
    fv["Annual_Premium"] = req.annual_premium
    fv["Policy_Sales_Channel"] = req.policy_sales_channel
    fv["Vintage"] = req.vintage
    # One-hot features
    gender_col = f"Gender_{req.gender}"
    if gender_col in fv:
        fv[gender_col] = 1
    vehicle_age_col = f"Vehicle_Age_{req.vehicle_age}"
    if vehicle_age_col in fv:
        fv[vehicle_age_col] = 1
    vehicle_damage_col = f"Vehicle_Damage_{req.vehicle_damage}"
    if vehicle_damage_col in fv:
        fv[vehicle_damage_col] = 1
    # Return as 2D array
    return np.array([fv[col] for col in FEATURE_COLUMNS])

# --- App Initialization ---
app = FastAPI(title="Insurance Policy Renewal Predictor")

try:
    bagging_model = load_model(BAGGING_MODEL_PATH)
    boosting_model = load_model(BOOSTING_MODEL_PATH)
except Exception as e:
    raise RuntimeError(str(e))

# --- API Endpoints ---
@app.get("/", tags=["Health"])
def read_root():
    """Health check endpoint."""
    return {"message": "Insurance Policy Renewal Prediction API"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(req: PolicyRequest):
    """Predict insurance policy renewal."""
    x = make_feature_vector(req)
    model = bagging_model if req.model_type == "bagging" else boosting_model
    try:
        proba = float(model.predict_proba([x])[0, 1])
        pred = int(model.predict([x])[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    return PredictionResponse(model=req.model_type, prediction=pred, probability=proba)


# To run the app, use the command:
# uvicorn app:app --host