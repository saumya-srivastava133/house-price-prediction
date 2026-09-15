from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from io import BytesIO
from pathlib import Path

import pandas as pd
import joblib


# --------------------------------------------------
# Model path
# --------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parent / "house_price_prediction_pipeline.joblib"

DROP_COLUMNS = [
    "Id",
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence"
]


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting residential house prices using a trained ML pipeline.",
    version="1.0.0"
)


# --------------------------------------------------
# Input schema
# --------------------------------------------------

class HouseFeatures(BaseModel):
    data: dict


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

def load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="Trained model not found"
        )

    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "House Price Prediction API is running"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(payload: HouseFeatures):

    model = load_model()

    X_new = pd.DataFrame([payload.data])

    X_new = X_new.drop(
        columns=DROP_COLUMNS,
        errors="ignore"
    )

    prediction = float(model.predict(X_new)[0])

    return {
        "predicted_sale_price": round(prediction, 2)
    }


# --------------------------------------------------
# Retraining endpoint
# --------------------------------------------------

@app.post("/retrain")
def retrain(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Upload a CSV file"
        )

    data = pd.read_csv(
        BytesIO(file.file.read())
    )

    if "SalePrice" not in data.columns:
        raise HTTPException(
            status_code=400,
            detail="CSV must contain SalePrice for supervised retraining"
        )

    X_new = data.drop(
        columns=["SalePrice"] + DROP_COLUMNS,
        errors="ignore"
    )

    y_new = data["SalePrice"]

    model = load_model()

    model.fit(X_new, y_new)

    joblib.dump(
        model,
        MODEL_PATH
    )

    return {
        "message": "Model retrained successfully",
        "rows_used": len(data)
    }