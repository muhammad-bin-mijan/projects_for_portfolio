"""Configuration for Diabetes Risk Prediction Dataset"""

# Dataset paths
DATASET_PATH = r"Datasets/Diabetes_Risk_Prediction_dataset/diabetes_risk_prediction_dataset.csv"
MODEL_PATH = 'models/diabetes_model.pkl'

# Target and splitting
TARGET_COLUMN = 'diabetes_risk'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Columns with missing values
NUMERIC_COLS_MISSING = ['Age', 'Blood_Glucose', 'HbA1c', 'HDL', 'LDL', 'Triglycerides', 'Total_Cholesterol']
CATEGORICAL_COLS = ['Gender', 'Physical_Activity', 'Medication_Adherence']

# Model parameters
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 15,
    'random_state': RANDOM_STATE
}