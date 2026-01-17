import pandas as pd
import numpy as np
import time
import os
import sys
import glob
import warnings
import yaml
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stats_utils import ks_test, psi, mmd_rbf
from src.preprocessing import clean_data
from src.feature_engineering import feature_eng
from river.drift import ADWIN, PageHinkley
import requests
import joblib
import monitoring.config as config

warnings.filterwarnings("ignore")

# Resolve paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_PATH = os.path.join(BASE_DIR, "params.yaml")

# Testing mode: limit iterations
MAX_WINDOWS = 5


def load_params():
    with open(PARAMS_PATH, 'r') as f:
        return yaml.safe_load(f)


def get_dataset_config(dataset_type="credit"):
    """Get configuration for a specific dataset."""
    params = load_params()
    
    if dataset_type == "credit":
        data_config = params['data']['credit_frauds']
        reference_path = os.path.join(BASE_DIR, data_config.get('reference_path', 
            "data/raw/credit_frauds/creditcard.csv"))
        target_col = data_config.get('target_col', 'Class')
        experiment_name = data_config.get('experiment_name', 'credit-fraud-detection')
    else:
        data_config = params['data']['telco']
        reference_path = os.path.join(BASE_DIR, data_config.get('reference_path',
            "data/preprocessed/telco/cleaned_telco.csv"))
        target_col = data_config.get('target_col', 'Churn')
        experiment_name = data_config.get('experiment_name', 'telco-churn-detection')
    
    return {
        'reference_path': reference_path,
        'target_col': target_col,
        'experiment_name': experiment_name
    }


def load_latest_model(dataset_type="credit"):
    """Load the latest model from MLflow artifacts for a specific dataset."""
    from app.model import load_latest_model as load_model
    return load_model(dataset_type)


def preprocess_stream_data(stream_df, dataset_type="credit"):
    """Preprocess raw stream data using the same pipeline as training."""
    target_col = 'Class' if dataset_type == 'credit' else 'Churn'
    
    # Step 1: Clean data (handle missing values, convert types)
    cleaned = clean_data(stream_df.copy())
    
    # Step 2: Feature engineering (encoding, scaling)
    # If target column exists, use it; otherwise create a dummy
    if target_col not in cleaned.columns:
        cleaned[target_col] = 0  # Dummy target for preprocessing
    
    X, y = feature_eng(cleaned, target_col)
    
    # Add target back for concept drift detection if it existed
    if target_col in stream_df.columns:
        # Map string labels to numeric
        if stream_df[target_col].dtype == 'object':
            y = stream_df[target_col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}).fillna(0).astype(int)
        else:
            y = stream_df[target_col].fillna(0).astype(int)
        X[target_col] = y.values
    
    return X


def check_feature_drift(ref_vals, cur_vals):
    """Run KS-test and PSI for a single feature."""
    stat, p = ks_test(ref_vals, cur_vals)
    psi_val = psi(ref_vals, cur_vals)
    return stat, p, psi_val


def send_retrain_signal(reason):
    """Notify retrain endpoint when drift detected (silent fail if endpoint unavailable)."""
    payload = {"reason": str(reason)}
    try:
        r = requests.post(config.RETRAIN_WEBHOOK, json=payload, timeout=2)
        print(f"Retrain signal sent: {r.status_code}")
    except Exception:
        pass  # Endpoint not running, skip silently


def run_stream_monitor(dataset_type="telco", stream_path=None):
    """Simulate streaming data and detect drift for a specific dataset."""
    
    # Get dataset-specific configuration
    ds_config = get_dataset_config(dataset_type)
    reference_path = ds_config['reference_path']
    target_col = ds_config['target_col']
    
    print(f"\n{'='*60}")
    print(f"Drift Monitor - {dataset_type.upper()}")
    print(f"{'='*60}")
    print(f"Reference: {reference_path}")
    print(f"Target column: {target_col}")
    
    # Load reference data
    ref_df = pd.read_csv(reference_path)
    
    # Determine numeric columns (exclude target, time-like columns, and IDs)
    exclude_cols = [target_col, 'Time', 'customerID']
    # Include float, int, and bool types as numeric
    numeric_cols = [c for c in ref_df.columns if c not in exclude_cols and 
                    (ref_df[c].dtype in ['float64', 'int64', 'float32', 'int32', 'bool'] or 
                     ref_df[c].dtype.name == 'bool')]
    feature_cols = [c for c in ref_df.columns if c not in exclude_cols and c != target_col]
    
    print(f"Numeric features: {len(numeric_cols)}")
    
    # Sample reference window for MMD
    ref_window = ref_df[numeric_cols].dropna().sample(n=min(2000, len(ref_df)), random_state=42).values
    
    # Load model for concept drift
    model = load_latest_model(dataset_type)
    print(f"Model loaded: {model is not None}")
    
    # River concept drift detectors
    adwin = ADWIN()
    ph = PageHinkley()
    
    # Determine stream path
    if stream_path is None:
        if dataset_type == "credit":
            stream_path = 'drifted_creditcard.csv'
        elif dataset_type == "telco":
            stream_path = 'drifted_telco.csv'
        else:
            stream_path = 'data/preprocessed/telco/cleaned_telco.csv'
    
    if not os.path.exists(stream_path):
        print(f"Stream file not found: {stream_path}")
        print("Create a drifted data file or provide --stream-path argument")
        return
    
    print(f"Stream: {stream_path}")
    print("-" * 60)
    
    # Load and preprocess stream data
    raw_stream_df = pd.read_csv(stream_path)
    print(f"Raw stream data: {len(raw_stream_df)} rows, {len(raw_stream_df.columns)} columns")
    
    # Check if stream data needs preprocessing (has raw categorical columns)
    needs_preprocessing = raw_stream_df.select_dtypes(include='object').shape[1] > 1
    
    if needs_preprocessing:
        print("Preprocessing stream data...")
        stream_df = preprocess_stream_data(raw_stream_df, dataset_type)
        print(f"Preprocessed: {len(stream_df)} rows, {len(stream_df.columns)} columns")
    else:
        stream_df = raw_stream_df
    
    N = len(stream_df)
    
    windows_processed = 0
    for start in range(0, N - config.WINDOW_SIZE + 1, config.STEP):
        if MAX_WINDOWS and windows_processed >= MAX_WINDOWS:
            print(f"\n[Test mode] Stopped after {MAX_WINDOWS} windows.")
            break
        cur = stream_df.iloc[start : start + config.WINDOW_SIZE]
        alarms = []
        
        # Per-feature statistical tests
        for col in numeric_cols:
            if col not in cur.columns:
                continue
            ref_vals = ref_df[col].dropna()
            cur_vals = cur[col].dropna()
            
            # Convert boolean to int for statistical tests
            if ref_vals.dtype == 'bool' or ref_vals.dtype == bool:
                ref_vals = ref_vals.astype(int)
            if cur_vals.dtype == 'bool' or cur_vals.dtype == bool:
                cur_vals = cur_vals.astype(int)
            
            # Skip if non-numeric values present
            if cur_vals.dtype == 'object':
                try:
                    cur_vals = pd.to_numeric(cur_vals, errors='coerce').dropna()
                except:
                    continue
            
            ref_vals = ref_vals.values.astype(float)
            cur_vals = cur_vals.values.astype(float)
            
            if len(cur_vals) < 2:
                continue
            stat, p, psi_val = check_feature_drift(ref_vals, cur_vals)
            if p < config.KS_P_THRESHOLD or psi_val > config.PSI_THRESHOLD:
                alarms.append((col, round(stat, 4), round(p, 6), round(psi_val, 4)))
        
        # Multivariate MMD test
        try:
            common_numeric = [c for c in numeric_cols if c in cur.columns]
            cur_numeric = cur[common_numeric].dropna().values
            if cur_numeric.shape[0] > 10:
                mmd_val = mmd_rbf(ref_window[:, :cur_numeric.shape[1]], cur_numeric)
                if mmd_val > config.MMD_THRESHOLD:
                    alarms.append(("MMD", round(mmd_val, 6)))
        except Exception as e:
            print(f"MMD error: {e}")
        
        # Concept drift detection using model predictions
        label_col = target_col if target_col in cur.columns else None
        if model is not None and label_col:
            try:
                # Get features for prediction (exclude target)
                X_cur = cur[[c for c in feature_cols if c in cur.columns]].fillna(0).values
                preds = model.predict(X_cur)
                labels = cur[label_col].values
                
                # Filter out NaN labels
                valid_mask = ~pd.isna(labels)
                labels = labels[valid_mask]
                preds = preds[valid_mask]
                
                concept_drift_detected = False
                for y, yhat in zip(labels, preds):
                    err = 0 if int(y) == int(yhat) else 1
                    ph.update(err)
                    adwin.update(err)
                    if ph.drift_detected:
                        print(f"    PageHinkley detected CONCEPT DRIFT at window {start}")
                        send_retrain_signal(f"PageHinkley detected concept drift in {dataset_type}")
                        concept_drift_detected = True
                        break
                    if adwin.drift_detected:
                        print(f"    ADWIN detected CONCEPT DRIFT at window {start}")
                        send_retrain_signal(f"ADWIN detected concept drift in {dataset_type}")
                        concept_drift_detected = True
                        break
                
                # Also report error rate for this window
                if len(labels) > 0:
                    error_rate = np.mean([int(y) != int(yhat) for y, yhat in zip(labels, preds)])
                    if error_rate > 0.1:  # High error threshold
                        print(f"   Error rate: {error_rate:.2%} (elevated)")
            except Exception as e:
                print(f"Concept drift error: {e}")
        
        if alarms:
            drift_features = [a[0] for a in alarms]
            print(f"[Window {start}-{start+config.WINDOW_SIZE}] Data drift in {len(alarms)} features: {drift_features}")
            send_retrain_signal({"dataset": dataset_type, "feature_alarms": alarms})
        else:
            print(f"[Window {start}-{start+config.WINDOW_SIZE}] No significant drift detected")
        
        windows_processed += 1
        time.sleep(0.1)  # Faster for testing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run drift monitor for credit or telco data')
    parser.add_argument('--dataset', type=str, default='telco', choices=['credit', 'telco'],
                        help='Dataset to monitor: credit or telco')
    parser.add_argument('--stream-path', type=str, default=None,
                        help='Path to streaming data CSV')
    args = parser.parse_args()
    
    run_stream_monitor(dataset_type=args.dataset, stream_path=args.stream_path)
