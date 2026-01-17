# config/constants
WINDOW_SIZE = 500        # current window size
STEP = 100               # slide step
KS_P_THRESHOLD = 0.01    # KS p-value threshold for drift
PSI_THRESHOLD = 0.2      # PSI > 0.2 indicates moderate drift
MMD_THRESHOLD = 0.001    # tune per dataset
RETRAIN_WEBHOOK = "http://localhost:8000/retrain"  # webhook to trigger model retraining