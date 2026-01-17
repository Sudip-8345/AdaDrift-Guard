# Self-Healing MLOps Pipeline

A production-ready self-healing MLOps system for binary classification tasks with automatic drift detection and model retraining capabilities. Supports multiple datasets including credit card fraud detection and telco customer churn prediction.

## Overview

This project implements an end-to-end MLOps pipeline that monitors deployed models for data and concept drift, automatically triggering retraining when performance degrades. The system uses statistical tests to detect distribution shifts and maintains model performance over time.

### Key Features

- Multi-dataset support (Credit Card Fraud, Telco Churn)
- Real-time prediction via REST API
- Data drift detection using KS-test, PSI, and MMD
- Concept drift monitoring with ADWIN and PageHinkley algorithms
- Automated retraining pipeline
- Experiment tracking with MLflow
- Interactive dashboard with Streamlit

## Supported Datasets

### Credit Card Fraud Detection
- Binary classification: Fraud (1) vs Not Fraud (0)
- 30 features (V1-V28, Time, Amount)
- Highly imbalanced dataset

### Telco Customer Churn
- Binary classification: Churn (1) vs Not Churn (0)
- Customer demographic and service features
- Moderate class imbalance

## Project Structure

```
Self-Healing-MLOps/
├── app/
│   ├── api.py              # FastAPI REST endpoints
│   ├── model.py            # Model wrapper and loading utilities
│   ├── streamlit_app.py    # Interactive dashboard
│   └── Dockerfile          # Container configuration
├── data/
│   ├── raw/                # Original datasets
│   ├── preprocessed/       # Cleaned data
│   └── training/           # Train/test splits
├── monitoring/
│   ├── config.py           # Drift detection thresholds
│   ├── drift_monitor.py    # Main monitoring logic
│   └── evidently_report.py # HTML drift reports
├── src/
│   ├── data_ingestion.py   # Data loading
│   ├── preprocessing.py    # Data cleaning
│   ├── feature_engineering.py
│   └── train.py            # Model training
├── utils/
│   └── stats_utils.py      # Statistical test implementations
├── mlruns/                 # MLflow experiment artifacts
├── params.yaml             # Training configuration
├── requirements.txt
└── docker-compose.yml
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Sudip-8345/AdaDrift-Guard.git
cd AdaDrift-Guard
```

2. Create a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # Linux/Mac
myenv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Training parameters are managed through `params.yaml`:

```yaml
data:
  credit_frauds:
    x_path: "data/training/credit_frauds/x_credit.csv"
    y_path: "data/training/credit_frauds/y_credit.csv"
    reference_path: "data/raw/credit_frauds/creditcard.csv"
    target_col: "Class"
  telco:
    x_path: "data/training/telco/x_telco.csv"
    y_path: "data/training/telco/y_telco.csv"
    reference_path: "data/preprocessed/telco/cleaned_telco.csv"
    target_col: "Churn"

model:
  type: "RandomForest"
  n_estimators: 100
  class_weight: "balanced"

drift:
  ks_p_threshold: 0.01
  psi_threshold: 0.2
  mmd_threshold: 0.001
```

Modify these values to adjust model behavior and drift sensitivity.

## Usage
### Run pipeline
```bash
dvc repro
```
or
### Training Models

Run the training script to train models for both datasets:

```bash
python src/train.py
```

This trains both the credit fraud and telco churn models. Each model is logged to its own MLflow experiment.

### Starting the API Server

Launch the FastAPI server:

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

API endpoints:
- `GET /health` - Health check (shows status of both models)
- `POST /predict` - Make predictions (specify dataset: "credit" or "telco")
- `POST /retrain` - Trigger model retraining

### Running the Dashboard

Start the Streamlit interface:

```bash
streamlit run app/streamlit_app.py
```

The dashboard provides:
- Dataset selector (Credit Card Fraud / Telco Churn)
- Prediction interface for both datasets
- Drift analysis tools
- Model performance metrics

### Monitoring for Drift

Run the drift monitor for a specific dataset:

```bash
# For credit card fraud
python monitoring/drift_monitor.py --dataset credit --stream-path drifted_creditcard.csv

# For telco churn
python monitoring/drift_monitor.py --dataset telco --stream-path drifted_telco.csv
```

Generate an Evidently report:

```bash
# For credit card fraud
python monitoring/evidently_report.py --dataset credit

# For telco churn
python monitoring/evidently_report.py --dataset telco
```

## Docker Deployment

Build and run with Docker Compose:

```bash
docker-compose up --build
```

Or build the API container separately:

```bash
cd app
docker build -t mlops-api .
docker run -p 8000:8000 mlops-api
```

## Drift Detection Methods

### Data Drift

The system monitors feature distributions using:

- **KS-test (Kolmogorov-Smirnov)**: Non-parametric test comparing cumulative distributions. A p-value below 0.01 indicates significant drift.

- **PSI (Population Stability Index)**: Measures how much a variable's distribution has shifted. Values above 0.2 suggest substantial change.

- **MMD (Maximum Mean Discrepancy)**: Kernel-based method for multivariate drift detection across all features simultaneously.

### Concept Drift

Model performance degradation is tracked using:

- **ADWIN (Adaptive Windowing)**: Maintains a variable-length window of recent predictions, detecting changes in error rate.

- **PageHinkley**: Sequential analysis method that detects changes in the mean of a sequence of observations.

## API Reference

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "credit_model_loaded": true,
  "telco_model_loaded": true
}
```

### Predict Endpoint

For credit card fraud:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {"V1": -1.35, "V2": 0.45, ...}, "dataset": "credit"}'
```

For telco churn:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {"tenure": 12, "MonthlyCharges": 50.5, ...}, "dataset": "telco"}'
```

Response:
```json
{
  "dataset": "credit",
  "pred": 0,
  "proba_negative": 0.9977,
  "proba_positive": 0.0023,
  "threshold": 0.5,
  "interpretation": "Not Fraud"
}
```

### Retrain Endpoint

```bash
curl -X POST http://localhost:8000/retrain \
  -H "Content-Type: application/json" \
  -d '{"reason": "Drift detected", "dataset": "credit"}'
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
flake8 src/ app/ monitoring/
black src/ app/ monitoring/ --check
```

## Troubleshooting

**API Connection Error**

Make sure the API server is running on port 8000. Check with:
```bash
curl http://localhost:8000/health
```

**Model Not Found**

Ensure you have trained at least one model:
```bash
python src/train.py
```

**Import Errors**

Verify all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- Credit card fraud dataset from Kaggle
- MLflow for experiment tracking
- Evidently for drift reporting
- River library for online learning algorithms
