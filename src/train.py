import os
import json
import yaml
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, RocCurveDisplay, ConfusionMatrixDisplay, classification_report
import warnings
import logging

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.ERROR)

# Load parameters
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_PATH = os.path.join(BASE_DIR, "params.yaml")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def load_params():
    with open(PARAMS_PATH, 'r') as f:
        return yaml.safe_load(f)

def load_data(path, name):
    full_path = os.path.join(BASE_DIR, path)
    if os.path.exists(full_path):
        return pd.read_csv(full_path)
    else:
        print(f"{name} data not found at {full_path}")
        return None
    
def train_and_log(X, Y, params, output_dir='plots', model_name='model'):
    model_params = params['model']
    train_params = params['training']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, 
        test_size=train_params['test_size'], 
        random_state=train_params['random_state']
    )
    
    model = RandomForestClassifier(
        n_estimators=model_params['n_estimators'],
        max_depth=model_params['max_depth'],
        min_samples_split=model_params['min_samples_split'],
        min_samples_leaf=model_params['min_samples_leaf'],
        class_weight=model_params['class_weight'],
        random_state=model_params['random_state'],
        n_jobs=model_params['n_jobs']
    )
    
    model.fit(X_train, y_train.values.ravel())
    y_preds = model.predict_proba(X_test)[:, 1]
    y_pred_classes = model.predict(X_test)
    auc = roc_auc_score(y_test, y_preds)
    report = classification_report(y_test, y_pred_classes, output_dict=True)
    
    full_output_dir = os.path.join(BASE_DIR, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)
    roc_path = os.path.join(full_output_dir, "roc.png")
    cm_path = os.path.join(full_output_dir, "cm.png")
    cr_path = os.path.join(full_output_dir, "clf_report.json")
    
    plt.figure()
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.savefig(roc_path)
    plt.close()

    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_classes)
    plt.title("Confusion Matrix")
    plt.savefig(cm_path)
    plt.close()
    
    with open(cr_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # Save model locally to dedicated file
    os.makedirs(MODELS_DIR, exist_ok=True)
    local_model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    joblib.dump(model, local_model_path)
    print(f"Model saved locally to: {local_model_path}")
        
    mlflow_params = params['mlflow']
    mlflow.set_experiment(mlflow_params['experiment_name'])
    
    with mlflow.start_run():
        mlflow.log_param("model", model_params['type'])
        mlflow.log_param("n_estimators", model_params['n_estimators'])
        mlflow.log_param("class_weight", model_params['class_weight'])
        mlflow.log_metric("auc", float(auc))
        mlflow.log_metric("precision", float(report['1']['precision']))
        mlflow.log_metric("recall", float(report['1']['recall']))
        mlflow.log_metric("f1", float(report['1']['f1-score']))
        mlflow.log_artifact(cm_path, artifact_path="plots")
        mlflow.log_artifact(roc_path, artifact_path="plots")
        mlflow.log_dict(report, "classification_report.json")
        mlflow.sklearn.log_model(model, artifact_path="model")
        
    print(f"Training complete. AUC: {auc:.4f}")
    print(f"Precision: {report['1']['precision']:.4f}, Recall: {report['1']['recall']:.4f}, F1: {report['1']['f1-score']:.4f}")
    return model, auc
    

if __name__ == "__main__":
    params = load_params()
    
    # Train credit fraud model
    credit_data = params['data']['credit_frauds']
    x_credit = load_data(credit_data['x_path'], "X_credit")
    y_credit = load_data(credit_data['y_path'], "Y_credit")
    
    if x_credit is not None and y_credit is not None:
        print("Training Credit Card Fraud Detection Model...")
        credit_params = params.copy()
        credit_params['mlflow']['experiment_name'] = 'credit-fraud-detection'
        train_and_log(x_credit, y_credit, credit_params, output_dir=credit_data['output_dir'], model_name='credit_model')
    
    # Train telco churn model
    telco_data = params['data']['telco']
    x_telco = load_data(telco_data['x_path'], "X_telco")
    y_telco = load_data(telco_data['y_path'], "Y_telco")
    
    if x_telco is not None and y_telco is not None:
        print("\nTraining Telco Churn Model...")
        telco_params = params.copy()
        telco_params['mlflow']['experiment_name'] = 'telco-churn-detection'
        train_and_log(x_telco, y_telco, telco_params, output_dir=telco_data['output_dir'], model_name='telco_model')