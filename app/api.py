from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import subprocess
import os
import sys
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model import ModelWrapper, load_latest_model
from src.preprocessing import clean_data
from src.feature_engineering import feature_eng

app = FastAPI(title="Self-Healing MLOps API")

# Resolve paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(BASE_DIR, "src", "train.py")

# Load models for both datasets on startup
models = {
    "credit": load_latest_model("credit"),
    "telco": load_latest_model("telco")
}

for name, model in models.items():
    if model is None:
        print(f"Warning: No {name} model loaded.")
    else:
        print(f"{name.capitalize()} model loaded: {type(model.model).__name__}")


class PredictRequest(BaseModel):
    data: dict
    dataset: Optional[str] = "credit"  # "credit" or "telco"


class RetrainRequest(BaseModel):
    reason: str = None
    dataset: Optional[str] = None  # None = retrain all


@app.get("/health")
def health():
    return {
        "status": "healthy", 
        "credit_model_loaded": models["credit"] is not None and models["credit"].model is not None,
        "telco_model_loaded": models["telco"] is not None and models["telco"].model is not None
    }


@app.post("/predict")
def predict(req: PredictRequest):
    dataset = req.dataset if req.dataset in models else "credit"
    model = models.get(dataset)
    
    if model is None or model.model is None:
        return {"error": f"No {dataset} model loaded"}
    
    try:
        df = pd.DataFrame([req.data])
        
        # For telco, preprocess raw data to match model features
        if dataset == "telco":
            # Check if data needs preprocessing (has string columns)
            if df.select_dtypes(include='object').shape[1] > 0:
                # Add dummy target for preprocessing
                df['Churn'] = 0
                cleaned = clean_data(df.copy())
                X, _ = feature_eng(cleaned, 'Churn')
                
                # Get expected features from model
                expected_features = model.model.feature_names_in_ if hasattr(model.model, 'feature_names_in_') else None
                
                if expected_features is not None:
                    # Add missing columns with 0
                    for col in expected_features:
                        if col not in X.columns:
                            X[col] = 0
                    # Reorder to match model
                    X = X[expected_features]
                
                df = X
        
        # Get raw probabilities from model (both classes)
        raw_proba = model.model.predict_proba(df.values)
        proba_negative = float(raw_proba[0][0])  # Class 0 (Not Fraud / Not Churn)
        proba_positive = float(raw_proba[0][1])  # Class 1 (Fraud / Churn)
        
        # Threshold-based prediction
        threshold = 0.5
        pred = 1 if proba_positive > threshold else 0
        
        # Dataset-specific labels
        if dataset == "credit":
            label_positive = "Fraud"
            label_negative = "Not Fraud"
        else:
            label_positive = "Churn"
            label_negative = "Not Churn"
        
        return {
            "dataset": dataset,
            "pred": pred,
            "proba_negative": proba_negative,
            "proba_positive": proba_positive,
            "threshold": threshold,
            "interpretation": label_positive if pred == 1 else label_negative
        }
    except Exception as e:
        return {"error": str(e)}


def run_training_subprocess():
    """Run training script and reload models after completion."""
    global models
    try:
        proc = subprocess.run(
            ["python", TRAIN_SCRIPT],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )
        if proc.returncode == 0:
            print("Training completed successfully")
            # Reload both models after training
            models["credit"] = load_latest_model("credit")
            models["telco"] = load_latest_model("telco")
            print("Models reloaded")
        else:
            print(f"Training failed: {proc.stderr}")
        return proc.returncode
    except Exception as e:
        print(f"Training error: {e}")
        return -1


@app.post("/retrain")
def retrain(req: RetrainRequest, background_tasks: BackgroundTasks):
    """Trigger model retraining in background."""
    background_tasks.add_task(run_training_subprocess)
    return {"status": "retrain started", "reason": req.reason, "dataset": req.dataset or "all"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
