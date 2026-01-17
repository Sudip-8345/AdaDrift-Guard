import kagglehub
import shutil
import os

def ingest_data():
    try:
        # 2. Define your target local folder
        telco_folder = "./data/raw/telco_data"
        credit_folder = "./data/raw/credit_frauds"
        
        os.makedirs(telco_folder, exist_ok=True)
        os.makedirs(credit_folder, exist_ok=True)
        
        # Check if files already exist
        telco_file = os.path.join(telco_folder, "telco_churn.csv")
        credit_file = os.path.join(credit_folder, "creditcard.csv")
        
        # Only download if files don't exist
        if not os.path.exists(telco_file) or not os.path.exists(credit_file):
            print("Downloading datasets...")
            telco_path = kagglehub.dataset_download("blastchar/telco-customer-churn")
            credit_frauds_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        else:
            print("Files already exist, skipping download...")
            return

        # 3. Copy files for Telco Data
        print(f"Telco source path: {telco_path}")
        telco_files = os.listdir(telco_path)
        print(f"Telco files found: {telco_files}")
        
        for filename in telco_files:
            target_name = "telco_churn.csv" if filename == "WA_Fn-UseC_-Telco-Customer-Churn.csv" else filename
            src = os.path.join(telco_path, filename)
            dst = os.path.join(telco_folder, target_name)
            shutil.copy2(src, dst)
            print(f"Copied: {filename} -> {dst}")
        
        print(f"Successfully copied Telco data to: {telco_folder}")

        # 4. Copy files for Credit Fraud Data
        print(f"Credit source path: {credit_frauds_path}")
        credit_files = os.listdir(credit_frauds_path)
        print(f"Credit files found: {credit_files}")
        
        for filename in credit_files:
            src = os.path.join(credit_frauds_path, filename)
            dst = os.path.join(credit_folder, filename)
            shutil.copy2(src, dst)
            print(f"Copied: {filename} -> {dst}")

        print(f"Successfully copied Credit Fraud data to: {credit_folder}")

    except PermissionError:
        print("Error: Permission denied. Please check your folder write permissions.")
    except FileNotFoundError as e:
        print(f"Error: A file or directory was not found: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    ingest_data()