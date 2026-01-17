import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_data(path, name):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print(f"{name} data not found at {path}")
        return None

def feature_eng(df, target_col):
    if df is None:
        return None, None

    # ----- TARGET -----
    if df[target_col].dtype == 'object':
        y = df[target_col].map({'Yes': 1, 'No': 0})
    else:
        y = df[target_col].astype(int)

    # ----- DROP TARGET + ID -----
    X = df.drop(columns=[target_col, 'customerID'], errors='ignore')

    # ----- HANDLE OBJECT FEATURES -----
    cat_cols = X.select_dtypes(include='object').columns.tolist()

    # Binary categorical mapping
    binary_map = {
        'Yes': 1, 'No': 0,
        'Male': 1, 'Female': 0
    }

    for col in cat_cols:
        if X[col].nunique() == 2:
            X[col] = X[col].map(binary_map)

    # One-hot encode remaining categorical columns
    X = pd.get_dummies(X, drop_first=True)

    # ----- NUMERIC SCALING -----
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y


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
    TELCO_preprocessed = "data/preprocessed/telco/cleaned_telco.csv"
    target_telco = 'Churn'
    CREDIT_preprocessed = "data/preprocessed/credit_frauds/cleaned_credit.csv"
    target_credit = 'Class'
    training_dir = './data/training'

    # Engineered Telco
    preprocessed_telco = load_data(TELCO_preprocessed, "Telco")
    x_telco, y_telco = feature_eng(preprocessed_telco, target_telco)
    save_data(x_telco, f"{training_dir}/telco", "x_telco.csv")
    save_data(y_telco, f"{training_dir}/telco", "y_telco.csv")
    
    # Engineered Credit
    preprocessed_credit = load_data(CREDIT_preprocessed, "Credit Card")
    x_credit, y_credit = feature_eng(preprocessed_credit, target_credit)
    save_data(x_credit, f"{training_dir}/credit_frauds", "x_credit.csv")
    save_data(y_credit, f"{training_dir}/credit_frauds", "y_credit.csv")