from evidently import Report
from evidently import Dataset, DataDefinition, BinaryClassification
from evidently.presets import DataDriftPreset, ClassificationPreset
import pandas as pd
import os
import yaml
import argparse

# Resolve paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_PATH = os.path.join(BASE_DIR, "params.yaml")


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
        output_dir = os.path.join(BASE_DIR, data_config.get('output_dir', 'plots/credit_frauds'))
    else:
        data_config = params['data']['telco']
        reference_path = os.path.join(BASE_DIR, data_config.get('reference_path',
            "data/preprocessed/telco/cleaned_telco.csv"))
        target_col = data_config.get('target_col', 'Churn')
        output_dir = os.path.join(BASE_DIR, data_config.get('output_dir', 'plots/telco'))
    
    return {
        'reference_path': reference_path,
        'target_col': target_col,
        'output_dir': output_dir,
        'report_path': os.path.join(output_dir, 'evidently_report.html')
    }


def create_evidently_report(reference_df, current_df, target_col="Class", output_path=None):
    """Generate Evidently data drift and classification report."""
    if output_path is None:
        output_path = os.path.join(BASE_DIR, "plots", "evidently_report.html")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Determine if we have target/prediction columns
    has_target = target_col in reference_df.columns
    pred_col = "prediction" if "prediction" in current_df.columns else None
    
    # Choose metrics based on available columns
    metrics = [DataDriftPreset()]
    if has_target and pred_col:
        metrics.append(ClassificationPreset())
    
    report = Report(metrics=metrics)
    
    # Define the schema using new Evidently API
    if has_target and pred_col:
        data_definition = DataDefinition(
            classification=[BinaryClassification(target=target_col, prediction=pred_col)]
        )
    else:
        data_definition = DataDefinition()

    # Wrap dataframes into Dataset objects
    curr_dataset = Dataset.from_pandas(current_df, data_definition=data_definition)
    ref_dataset = Dataset.from_pandas(reference_df, data_definition=data_definition)

    # Run and save the report (run returns a Snapshot)
    snapshot = report.run(reference_data=ref_dataset, current_data=curr_dataset)
    snapshot.save_html(output_path)
    print(f"Evidently report saved: {output_path}")
    return snapshot


def run_report(dataset_type="credit", stream_path=None):
    """Generate drift report for a specific dataset."""
    ds_config = get_dataset_config(dataset_type)
    
    print(f"\n{'='*60}")
    print(f"Evidently Report - {dataset_type.upper()}")
    print(f"{'='*60}")
    print(f"Reference: {ds_config['reference_path']}")
    print(f"Target column: {ds_config['target_col']}")
    
    # Load reference data
    ref = pd.read_csv(ds_config['reference_path']).sample(n=2000, random_state=42)
    
    # Load current/stream data
    if stream_path is None:
        if dataset_type == "credit":
            stream_path = 'drifted_creditcard.csv'
        else:
            stream_path = 'drifted_telco.csv'
    
    if not os.path.exists(stream_path):
        print(f"Stream file not found: {stream_path}")
        print("Using reference data sample as current (no drift report)")
        cur = ref.sample(n=min(1000, len(ref)), random_state=123)
    else:
        print(f"Current: {stream_path}")
        cur = pd.read_csv(stream_path).head(1000)
    
    # Align columns (current may have different columns)
    common_cols = list(set(ref.columns) & set(cur.columns))
    ref = ref[common_cols]
    cur = cur[common_cols]
    
    create_evidently_report(
        ref, cur, 
        target_col=ds_config['target_col'],
        output_path=ds_config['report_path']
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Evidently drift report')
    parser.add_argument('--dataset', type=str, default='credit', choices=['credit', 'telco'],
                        help='Dataset to analyze: credit or telco')
    parser.add_argument('--stream-path', type=str, default=None,
                        help='Path to streaming/current data CSV')
    args = parser.parse_args()
    
    run_report(dataset_type=args.dataset, stream_path=args.stream_path)
