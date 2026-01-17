import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stats_utils import ks_test, psi, mmd_rbf
from app.model import load_latest_model
from src.preprocessing import clean_data
from src.feature_engineering import feature_eng

API_URL = "http://localhost:8000"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_PATH = os.path.join(BASE_DIR, "params.yaml")


def load_params():
    with open(PARAMS_PATH, 'r') as f:
        return yaml.safe_load(f)


def get_dataset_config(dataset_type):
    """Get configuration for a specific dataset."""
    params = load_params()
    
    if dataset_type == "Credit Card Fraud":
        data_config = params['data']['credit_frauds']
        reference_path = os.path.join(BASE_DIR, data_config.get('reference_path', 
            "data/raw/credit_frauds/creditcard.csv"))
        target_col = data_config.get('target_col', 'Class')
        api_dataset = "credit"
        positive_label = "Fraud"
        negative_label = "Not Fraud"
    else:
        data_config = params['data']['telco']
        # Use cleaned_telco for UI (has target column) instead of x_telco
        reference_path = os.path.join(BASE_DIR, "data/preprocessed/telco/cleaned_telco.csv")
        target_col = data_config.get('target_col', 'Churn')
        api_dataset = "telco"
        positive_label = "Churn"
        negative_label = "Not Churn"
    
    return {
        'reference_path': reference_path,
        'target_col': target_col,
        'api_dataset': api_dataset,
        'positive_label': positive_label,
        'negative_label': negative_label
    }


st.set_page_config(page_title="Self-Healing MLOps Dashboard", layout="wide")
st.title("Self-Healing MLOps Dashboard")

# Dataset selector in sidebar
with st.sidebar:
    st.header("Configuration")
    dataset_choice = st.selectbox(
        "Select Dataset",
        ["Credit Card Fraud", "Telco Churn"],
        key="dataset_selector"
    )
    
    ds_config = get_dataset_config(dataset_choice)
    st.write(f"Target: {ds_config['target_col']}")
    st.write("---")
    
    st.header("About")
    st.write("Self-Healing MLOps system for classification tasks.")
    st.write("---")
    st.write("Features:")
    st.write("- Real-time prediction")
    st.write("- Data drift detection (KS, PSI, MMD)")
    st.write("- Concept drift monitoring")
    st.write("- Auto-retrain trigger")
    st.write("---")
    
    st.write("API Status:")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        if resp.status_code == 200:
            health = resp.json()
            st.success("Connected")
            credit_status = "OK" if health.get('credit_model_loaded') else "Missing"
            telco_status = "OK" if health.get('telco_model_loaded') else "Missing"
            st.write(f"Credit Model: {credit_status}")
            st.write(f"Telco Model: {telco_status}")
        else:
            st.error("Error")
    except:
        st.error("Offline")


@st.cache_data
def load_reference(path, target_col):
    df = pd.read_csv(path)
    # Encode string labels to numeric if needed
    if target_col in df.columns:
        if df[target_col].dtype == 'object':
            df[target_col] = df[target_col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}).fillna(0).astype(int)
    return df


# Load reference data based on selected dataset
target_col = ds_config['target_col']
ref_df = load_reference(ds_config['reference_path'], target_col)

# Determine feature columns
exclude_cols = [target_col, 'customerID']
feature_cols = [c for c in ref_df.columns if c not in exclude_cols]
numeric_cols = [c for c in feature_cols if ref_df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

tab1, tab2 = st.tabs(["Prediction", "Drift Detection"])

# ==================== TAB 1: PREDICTION ====================
with tab1:
    st.header(f"Predict - {dataset_choice}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Features")
        
        sample_type = st.radio(
            "Sample type", 
            [f"Negative ({ds_config['negative_label']})", 
             f"Positive ({ds_config['positive_label']})", 
             "Manual Input"]
        )
        
        if "Negative" in sample_type:
            neg_df = ref_df[ref_df[target_col] == 0].head(10)
            if len(neg_df) == 0:
                st.warning("No negative samples found")
                sample = {}
            else:
                sample_idx = st.selectbox("Select negative sample", options=range(len(neg_df)))
                sample = neg_df.iloc[sample_idx][feature_cols].to_dict()
                st.info(f"{ds_config['negative_label']} samples typically show low positive probability")
        elif "Positive" in sample_type:
            pos_df = ref_df[ref_df[target_col] == 1].head(20).reset_index(drop=True)
            if len(pos_df) == 0:
                st.warning("No positive samples found")
                sample = {}
            else:
                sample_idx = st.selectbox("Select positive sample", options=range(len(pos_df)))
                sample = pos_df.iloc[sample_idx][feature_cols].to_dict()
                st.warning(f"{ds_config['positive_label']} samples show varied probabilities")
        else:
            sample = {}
            with st.expander("Enter feature values"):
                for col in feature_cols[:20]:  # Limit to 20 for UI
                    default_val = float(ref_df[col].mean()) if ref_df[col].dtype in ['float64', 'int64'] else 0.0
                    sample[col] = st.number_input(col, value=default_val, format="%.6f")
        
        with st.expander("View sample data"):
            st.json(sample)
    
    with col2:
        st.subheader("Prediction Result")
        if st.button("Predict", type="primary", key="predict_btn"):
            if not sample:
                st.warning("No sample selected")
            else:
                try:
                    resp = requests.post(
                        f"{API_URL}/predict", 
                        json={"data": sample, "dataset": ds_config['api_dataset']}, 
                        timeout=10
                    )
                    result = resp.json()
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        pred = result.get("pred", 0)
                        proba_pos = result.get("proba_positive", 0)
                        proba_neg = result.get("proba_negative", 1)
                        threshold = result.get("threshold", 0.5)
                        interpretation = result.get("interpretation", "Unknown")
                        
                        if pred == 1:
                            st.error(f"{ds_config['positive_label'].upper()} DETECTED: {interpretation}")
                        else:
                            st.success(f"{ds_config['negative_label'].upper()}: {interpretation}")
                        
                        st.write("---")
                        st.write("Raw Model Probabilities:")
                        
                        col_a, col_b = st.columns(2)
                        col_a.metric(f"P({ds_config['negative_label']})", f"{proba_neg:.6f}")
                        col_b.metric(f"P({ds_config['positive_label']})", f"{proba_pos:.6f}")
                        
                        st.write("Confidence Distribution:")
                        st.progress(proba_pos)
                        st.caption(f"Positive probability: {proba_pos:.6f} | Threshold: {threshold}")
                        
                        st.write("---")
                        st.write("Decision Logic:")
                        if proba_pos > threshold:
                            st.write(f"P(Positive) = `{proba_pos:.6f}` > threshold `{threshold}` -> {ds_config['positive_label']}")
                        else:
                            st.write(f"P(Positive) = `{proba_pos:.6f}` <= threshold `{threshold}` -> {ds_config['negative_label']}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("API not running. Start with: uvicorn app.api:app --reload")
                except Exception as e:
                    st.error(f"Error: {e}")

# ==================== TAB 2: DRIFT DETECTION ====================
with tab2:
    st.header(f"Data and Concept Drift Detection - {dataset_choice}")
    
    uploaded_file = st.file_uploader("Upload stream data CSV", type=['csv'], key="drift_uploader")
    
    if uploaded_file:
        stream_df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded: {len(stream_df)} rows, {len(stream_df.columns)} columns")
        
        # For telco, use preprocessed training data as reference for drift detection
        if ds_config['api_dataset'] == 'telco':
            # Load encoded reference data
            ref_encoded_path = os.path.join(BASE_DIR, "data/training/telco/x_telco.csv")
            ref_drift_df = pd.read_csv(ref_encoded_path)
            
            # Preprocess stream data
            stream_copy = stream_df.copy()
            if target_col in stream_copy.columns:
                pass
            else:
                stream_copy[target_col] = 0
            
            cleaned_stream = clean_data(stream_copy)
            stream_encoded, _ = feature_eng(cleaned_stream, target_col)
            
            # Get common columns
            drift_numeric_cols = [c for c in ref_drift_df.columns if c in stream_encoded.columns]
            st.write(f"Preprocessed stream: {len(stream_encoded)} rows, {len(drift_numeric_cols)} features")
        else:
            ref_drift_df = ref_df
            stream_encoded = stream_df
            drift_numeric_cols = numeric_cols
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Reference Data Stats")
            st.write(f"Rows: {len(ref_drift_df)}")
            common_numeric = [c for c in drift_numeric_cols if c in ref_drift_df.columns][:10]  # Show first 10
            if common_numeric:
                st.dataframe(ref_drift_df[common_numeric].describe().T[['mean', 'std', 'min', 'max']], height=300)
        
        with col2:
            st.subheader("Stream Data Stats")
            st.write(f"Rows: {len(stream_encoded)}")
            stream_numeric = [c for c in drift_numeric_cols if c in stream_encoded.columns][:10]
            if stream_numeric:
                st.dataframe(stream_encoded[stream_numeric].describe().T[['mean', 'std', 'min', 'max']], height=300)
        
        if st.button("Run Drift Analysis", type="primary", key="drift_btn"):
            st.subheader("Drift Analysis Results")
            
            common_cols = [c for c in drift_numeric_cols if c in stream_encoded.columns]
            drift_results = []
            for col in common_cols:
                ref_vals = ref_drift_df[col].dropna()
                cur_vals = stream_encoded[col].dropna()
                
                # Convert bool to float
                if ref_vals.dtype == 'bool':
                    ref_vals = ref_vals.astype(float)
                if cur_vals.dtype == 'bool':
                    cur_vals = cur_vals.astype(float)
                
                ref_vals = ref_vals.values.astype(float)
                cur_vals = cur_vals.values.astype(float)
                
                if len(cur_vals) < 2:
                    continue
                stat, p = ks_test(ref_vals, cur_vals)
                psi_val = psi(ref_vals, cur_vals)
                drift_detected = p < 0.01 or psi_val > 0.2
                drift_results.append({
                    'Feature': col,
                    'KS Statistic': round(stat, 4),
                    'KS p-value': round(p, 6),
                    'PSI': round(psi_val, 4),
                    'Drift': 'Yes' if drift_detected else 'No'
                })
            
            if drift_results:
                drift_df = pd.DataFrame(drift_results)
                drifted_count = (drift_df['Drift'] == 'Yes').sum()
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Features Analyzed", len(drift_results))
                c2.metric("Features with Drift", drifted_count, delta=f"{drifted_count/len(drift_results)*100:.1f}%")
                c3.metric("Drift Status", "DRIFT DETECTED" if drifted_count > len(drift_results)*0.3 else "OK")
                
                st.dataframe(
                    drift_df.style.applymap(
                        lambda x: 'background-color: #ffcccc' if x == 'Yes' else '', 
                        subset=['Drift']
                    ),
                    use_container_width=True
                )
                
                st.subheader("Multivariate Drift (MMD)")
                try:
                    ref_sample = ref_drift_df[common_cols].dropna().sample(n=min(2000, len(ref_drift_df)), random_state=42)
                    cur_sample = stream_encoded[common_cols].dropna().head(2000)
                    
                    # Convert bool columns to float
                    for col in ref_sample.columns:
                        if ref_sample[col].dtype == 'bool':
                            ref_sample[col] = ref_sample[col].astype(float)
                        if cur_sample[col].dtype == 'bool':
                            cur_sample[col] = cur_sample[col].astype(float)
                    
                    if len(cur_sample) > 10:
                        mmd_val = mmd_rbf(ref_sample.values.astype(float), cur_sample.values.astype(float))
                        mmd_drift = mmd_val > 0.001
                        
                        col_m1, col_m2 = st.columns(2)
                        col_m1.metric("MMD Value", f"{mmd_val:.6f}")
                        col_m2.metric("MMD Drift", "Yes" if mmd_drift else "No")
                except Exception as e:
                    st.warning(f"MMD calculation failed: {e}")
                
                st.subheader("Concept Drift Detection")
                label_col = target_col if target_col in stream_df.columns else None
                
                error_rate = 0
                if label_col:
                    model = load_latest_model(ds_config['api_dataset'])
                    if model:
                        try:
                            # Use already preprocessed stream data for telco
                            if ds_config['api_dataset'] == 'telco':
                                # Align features with model
                                expected_features = model.model.feature_names_in_ if hasattr(model.model, 'feature_names_in_') else None
                                X_processed = stream_encoded.copy()
                                if expected_features is not None:
                                    for col in expected_features:
                                        if col not in X_processed.columns:
                                            X_processed[col] = 0
                                    X_processed = X_processed[expected_features]
                                
                                preds = model.predict(X_processed)
                                # Get labels from original stream data
                                labels_raw = stream_df[label_col]
                                if labels_raw.dtype == 'object':
                                    labels = labels_raw.map({'Yes': 1, 'No': 0}).fillna(0).astype(int).values
                                else:
                                    labels = labels_raw.fillna(0).astype(int).values
                            else:
                                X = stream_df[[c for c in feature_cols if c in stream_df.columns]].fillna(0)
                                preds = model.predict(X)
                                labels = stream_df[label_col].fillna(0).astype(int).values
                            
                            errors = (preds != labels).astype(int)
                            error_rate = errors.mean()
                            
                            col_c1, col_c2, col_c3 = st.columns(3)
                            col_c1.metric("Samples", len(labels))
                            col_c2.metric("Error Rate", f"{error_rate:.2%}")
                            col_c3.metric("Concept Drift", "Likely" if error_rate > 0.1 else "OK")
                            
                            st.write("Error Distribution (rolling window):")
                            window = 100
                            rolling_error = pd.Series(errors).rolling(window).mean()
                            st.line_chart(rolling_error.dropna(), use_container_width=True)
                            
                        except Exception as e:
                            st.warning(f"Concept drift check failed: {e}")
                    else:
                        st.info("No model loaded for concept drift detection")
                else:
                    st.info(f"No '{target_col}' column in stream data - concept drift requires labels")
                
                st.subheader("Recommendation")
                if drifted_count > len(drift_results) * 0.3 or error_rate > 0.1:
                    st.warning("Significant drift detected. Consider retraining the model.")
                    if st.button("Trigger Retrain", key="retrain_btn"):
                        try:
                            resp = requests.post(
                                f"{API_URL}/retrain", 
                                json={"reason": "Manual trigger from UI", "dataset": ds_config['api_dataset']}
                            )
                            st.success(f"Retrain triggered: {resp.json()}")
                        except:
                            st.error("API not available")
                else:
                    st.success("No significant drift. Model is performing well.")
            else:
                st.warning("No common numeric features found for drift analysis")
