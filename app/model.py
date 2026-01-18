import joblib
import mlflow.sklearn
import numpy as np
import pandas as pd
import os
import glob

class ModelWrapper:
    def __init__(self, model_path=None):
        self.model = None
        self.preprocess = None
        if model_path:
            self.load(model_path)

    def load(self, model_path):
        """Load model from path - supports joblib or MLflow artifacts."""
        if os.path.isdir(model_path):
            # Check for joblib files
            if os.path.exists(os.path.join(model_path, "model.joblib")):
                self.model = joblib.load(os.path.join(model_path, "model.joblib"))
                preprocess_path = os.path.join(model_path, "preprocess.joblib")
                if os.path.exists(preprocess_path):
                    self.preprocess = joblib.load(preprocess_path)
            # Check for MLflow model.pkl
            elif os.path.exists(os.path.join(model_path, "model.pkl")):
                self.model = joblib.load(os.path.join(model_path, "model.pkl"))
            else:
                raise FileNotFoundError(f"No model found in {model_path}")
        elif model_path.endswith(".pkl") or model_path.endswith(".joblib"):
            self.model = joblib.load(model_path)
        else:
            raise ValueError(f"Unsupported model path: {model_path}")

    def predict_proba(self, X_df):
        """Return probability of positive class."""
        X = X_df if self.preprocess is None else self.preprocess.transform(X_df)
        if hasattr(X, 'values'):
            X = X.values
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X_df, threshold=0.5):
        """Return binary predictions."""
        proba = self.predict_proba(X_df)
        return (proba > threshold).astype(int)


def get_experiment_id(experiment_name):
    """Get MLflow experiment ID by name from meta.yaml files."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlruns_dir = os.path.join(base_dir, "mlruns")
    
    for exp_dir in glob.glob(os.path.join(mlruns_dir, "*")):
        if not os.path.isdir(exp_dir):
            continue
        meta_path = os.path.join(exp_dir, "meta.yaml")
        if os.path.exists(meta_path):
            try:
                import yaml
                with open(meta_path) as f:
                    meta = yaml.safe_load(f)
                if meta.get("name") == experiment_name:
                    return os.path.basename(exp_dir)
            except:
                pass
    return None


def load_latest_model(dataset_type="credit"):
    """Load the latest model from MLflow artifacts for a specific dataset.
    
    Args:
        dataset_type: 'credit' for credit fraud or 'telco' for telco churn
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Map dataset type to experiment name
    experiment_map = {
        "credit": "credit-fraud-detection",
        "telco": "telco-churn-detection"
    }
    experiment_name = experiment_map.get(dataset_type, "credit-fraud-detection")
    
    # Define expected feature patterns for each dataset
    telco_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    credit_features = ['Time', 'V1', 'V2', 'Amount']
    
    # First try to find the experiment ID by name
    exp_id = get_experiment_id(experiment_name)
    
    # Collect all model paths
    all_model_paths = []
    
    if exp_id:
        models_dir = os.path.join(base_dir, "mlruns", exp_id, "models")
        all_model_paths.extend(glob.glob(os.path.join(models_dir, "m-*", "artifacts", "model.pkl")))
    
    # Also search all experiments and filter by feature names
    for exp_dir in glob.glob(os.path.join(base_dir, "mlruns", "*")):
        if os.path.isdir(exp_dir):
            models_dir = os.path.join(exp_dir, "models")
            all_model_paths.extend(glob.glob(os.path.join(models_dir, "m-*", "artifacts", "model.pkl")))
    
    # Remove duplicates and filter by dataset type
    all_model_paths = list(set(all_model_paths))
    
    # Filter models by their feature names
    valid_models = []
    for model_path in all_model_paths:
        try:
            model = joblib.load(model_path)
            if hasattr(model, 'feature_names_in_'):
                features = list(model.feature_names_in_)
                if dataset_type == "telco" and any(f in features for f in telco_features):
                    valid_models.append(model_path)
                elif dataset_type == "credit" and any(f in features for f in credit_features):
                    valid_models.append(model_path)
        except:
            pass
    
    if not valid_models:
        return None
    
    latest = max(valid_models, key=os.path.getmtime)
    return ModelWrapper(latest)


if __name__ == "__main__":
    # Test loading both models
    for ds in ["credit", "telco"]:
        print(f"\n--- {ds.upper()} MODEL ---")
        wrapper = load_latest_model(ds)
        if wrapper and wrapper.model:
            print(f"Model loaded: {type(wrapper.model).__name__}")
            print(f"Features expected: {wrapper.model.n_features_in_}")
        else:
            print("No model found")