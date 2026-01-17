import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import os

def load_data(path, name):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print(f"{name} data not found at {path}")
        return None

def clean_data(df, time_col=None):
    if df is None: return None
    
    # Specific fix for Telco: Convert TotalCharges to numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Separate numeric and non-numeric to avoid Imputer errors on strings
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    
    si = SimpleImputer(strategy='median')
    
    # Impute only numeric data
    df_numeric = pd.DataFrame(si.fit_transform(df[numeric_cols]), 
                              columns=numeric_cols, 
                              index=df.index)
    
    # Combine back with non-numeric data
    df = pd.concat([df_numeric, df[non_numeric_cols]], axis=1)

    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)
    
    return df


def save_data(df, folder_path, filename):
    if df is not None:
        os.makedirs(folder_path, exist_ok=True)
        full_path = os.path.join(folder_path, filename)
        df.to_csv(full_path, index=False)
        print(f"Saved: {full_path}")
    else:
        print("Dataframe is empty, nothing to save.")

if __name__ == "__main__":
    # Paths
    TELCO_RAW = "data/raw/telco_data/telco_churn.csv"
    CREDIT_RAW = "data/raw/credit_frauds/creditcard.csv"
    PREPROCESSED_DIR = './data/preprocessed'

    # Process Telco
    raw_telco = load_data(TELCO_RAW, "Telco")
    cleaned_telco = clean_data(raw_telco)
    save_data(cleaned_telco, f"{PREPROCESSED_DIR}/telco", "cleaned_telco.csv")
    
    # Process Credit
    raw_credit = load_data(CREDIT_RAW, "Credit Card")
    cleaned_credit = clean_data(raw_credit)
    save_data(cleaned_credit, f"{PREPROCESSED_DIR}/credit_frauds", "cleaned_credit.csv")