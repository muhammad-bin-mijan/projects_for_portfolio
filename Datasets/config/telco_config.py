"""Configuration for Telco Customer Churn Dataset"""

# Dataset paths
DATASET_PATH = r"Datasets/Telco_Customer_Churn_dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = r"models/telco_model.pkl"

# Target and splitting
TARGET_COLUMN = "Churn"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Columns with missing or categorical values
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

# Columns to drop (like ID, etc.)
DROP_COLS = ["customerID"]

# Model parameters
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced"
}
