from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import subprocess
import os
import sys
import pandas as pd
import numpy as np
import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import clean_data
from src.feature_engineering import feature_eng

app = FastAPI(title="Self-Healing MLOps API")

# Resolve paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(BASE_DIR, "src", "train.py")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Define model paths for each dataset
MODEL_PATHS = {
    "credit": os.path.join(MODELS_DIR, "credit_model.joblib"),
    "telco": os.path.join(MODELS_DIR, "telco_model.joblib")
}

# Load RAW (unscaled) data statistics for proper scaling during prediction
TELCO_RAW_PATH = os.path.join(BASE_DIR, "data/preprocessed/telco/cleaned_telco.csv")
telco_scaling_stats = None
if os.path.exists(TELCO_RAW_PATH):
    _raw_df = pd.read_csv(TELCO_RAW_PATH)
    _numeric_cols = _raw_df.select_dtypes(include=[np.number]).columns.tolist()
    _numeric_cols = [c for c in _numeric_cols if c not in ['Churn']]
    telco_scaling_stats = {
        'mean': _raw_df[_numeric_cols].mean(),
        'std': _raw_df[_numeric_cols].std().replace(0, 1),
    }
    print(f"Loaded telco scaling stats from raw data: {_numeric_cols}")

# Load Credit Card raw data statistics for scaling
CREDIT_RAW_PATH = os.path.join(BASE_DIR, "data/raw/credit_frauds/creditcard.csv")
credit_scaling_stats = None
if os.path.exists(CREDIT_RAW_PATH):
    _raw_credit = pd.read_csv(CREDIT_RAW_PATH)
    _credit_cols = [c for c in _raw_credit.columns if c != 'Class']
    credit_scaling_stats = {
        'mean': _raw_credit[_credit_cols].mean(),
        'std': _raw_credit[_credit_cols].std().replace(0, 1),
    }
    print(f"Loaded credit scaling stats from raw data: {len(_credit_cols)} features")


def load_model(dataset_type):
    """Load model from dedicated local file."""
    model_path = MODEL_PATHS.get(dataset_type)
    if model_path and os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Loaded {dataset_type} model from {model_path}")
        return model
    print(f"Warning: No {dataset_type} model found at {model_path}")
    return None


# Load models for both datasets on startup
models = {
    "credit": load_model("credit"),
    "telco": load_model("telco")
}

for name, model in models.items():
    if model is None:
        print(f"Warning: No {name} model loaded.")
    else:
        print(f"{name.capitalize()} model loaded: {type(model).__name__}")


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
        "credit_model_loaded": models["credit"] is not None,
        "telco_model_loaded": models["telco"] is not None
    }


@app.post("/predict")
def predict(req: PredictRequest):
    dataset = req.dataset if req.dataset in models else "credit"
    model = models.get(dataset)
    
    if model is None:
        return {"error": f"No {dataset} model loaded"}
    
    try:
        df = pd.DataFrame([req.data])
        
        # For credit card, apply scaling
        if dataset == "credit":
            if credit_scaling_stats is not None:
                # Scale all features using raw data statistics
                for col in df.columns:
                    if col in credit_scaling_stats['mean'].index:
                        mean_val = credit_scaling_stats['mean'][col]
                        std_val = credit_scaling_stats['std'][col]
                        df[col] = (df[col] - mean_val) / std_val
        
        # For telco, preprocess raw data to match model features
        elif dataset == "telco":
            # Check if data needs preprocessing (has string columns OR raw feature names)
            has_object_cols = df.select_dtypes(include='object').shape[1] > 0
            # Also check if it has raw column names like 'gender', 'Partner' etc.
            raw_telco_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                            'PaperlessBilling', 'PaymentMethod']
            has_raw_cols = any(col in df.columns for col in raw_telco_cols)
            
            if has_object_cols or has_raw_cols:
                # Add dummy target for preprocessing
                df['Churn'] = 'No'
                
                # Convert TotalCharges to numeric if present
                if 'TotalCharges' in df.columns:
                    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
                    if df['TotalCharges'].isna().any():
                        df['TotalCharges'] = df['TotalCharges'].fillna(0)
                
                # Manual preprocessing instead of feature_eng to avoid wrong scaling
                X = df.drop(columns=['Churn', 'customerID'], errors='ignore')
                
                # Binary categorical mapping
                binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
                cat_cols = X.select_dtypes(include='object').columns.tolist()
                for col in cat_cols:
                    if X[col].iloc[0] in binary_map:
                        X[col] = X[col].map(binary_map)
                
                # One-hot encode remaining categorical columns
                X = pd.get_dummies(X, drop_first=True)
                
                # Get expected features from model
                expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
                
                if expected_features is not None:
                    # Add missing columns with 0
                    for col in expected_features:
                        if col not in X.columns:
                            X[col] = 0
                    # Reorder to match model
                    X = X[list(expected_features)]
                
                # Apply scaling using RAW data statistics (Z-score normalization)
                # Only scale the numeric columns that were scaled during training
                if telco_scaling_stats is not None:
                    numeric_to_scale = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
                    for col in numeric_to_scale:
                        if col in X.columns and col in telco_scaling_stats['mean'].index:
                            mean_val = telco_scaling_stats['mean'][col]
                            std_val = telco_scaling_stats['std'][col]
                            X[col] = (X[col] - mean_val) / std_val
                
                df = X
        
        # Get raw probabilities from model (both classes)
        raw_proba = model.predict_proba(df.values)
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
            # Reload both models after training from local files
            models["credit"] = load_model("credit")
            models["telco"] = load_model("telco")
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
