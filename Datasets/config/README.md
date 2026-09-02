# Dataset Configurations

This folder contains configuration files for each dataset in the portfolio project.

## Files

### `diabetes_config.py`
Configuration for Diabetes Risk Prediction dataset
- **Path**: `Datasets/Diabetes_Risk_Prediction_dataset/diabetes_risk_prediction_dataset.csv`
- **Target**: `diabetes_risk`
- **Rows**: 50,000 | **Columns**: 30+
- **Task**: Multi-class classification

### `telco_config.py`
Configuration for Telco Customer Churn dataset
- **Path**: `Datasets/Telco_Customer_Churn_dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Target**: `Churn`
- **Rows**: 7,043 | **Columns**: 21
- **Task**: Binary classification

### `titanic_config.py`
Configuration for Titanic Survival dataset
- **Path**: `Datasets/titanic/train.csv`
- **Target**: `Survived`
- **Rows**: 891 | **Columns**: 12
- **Task**: Binary classification

### `student_config.py`
Configuration for Student Performance dataset
- **Path**: `Datasets/Student_Performance/Students Performance Dataset.csv`
- **Target**: `Grade`
- **Rows**: 5,000 | **Columns**: 23
- **Task**: Classification/Regression

## Usage

Each config file contains:
- `DATASET_PATH` - Path to the CSV file
- `MODEL_PATH` - Where to save trained model
- `TARGET_COLUMN` - Column name of target variable
- `TEST_SIZE` - Train/test split ratio (default: 0.2)
- `RANDOM_STATE` - Random seed for reproducibility
- `NUMERIC_COLS` - List of numeric columns
- `CATEGORICAL_COLS` - List of categorical columns
- `DROP_COLS` - Columns to exclude from analysis
- `MODEL_PARAMS` - Default hyperparameters for model training

### Example Import

```python
from Datasets.config.diabetes_config import DATASET_PATH, TARGET_COLUMN, MODEL_PARAMS

# Use in notebook
df = pd.read_csv(DATASET_PATH)
model = train_model(X_train, y_train, algorithm='random_forest', params=MODEL_PARAMS)
```

## Structure

```
Datasets/config/
├── __init__.py                 # Package initialization
├── diabetes_config.py          # Diabetes dataset config
├── telco_config.py             # Telco Churn config
├── titanic_config.py           # Titanic config
└── student_config.py           # Student Performance config
```

## Best Practices

✅ All paths use raw strings (`r"..."`) to avoid escape issues
✅ Consistent across all datasets
✅ Easy to modify parameters without touching code
✅ Centralized configuration management

## Modifying Configurations

To update a configuration, edit the corresponding file:
- Change `TEST_SIZE` from 0.2 to 0.3
- Add/remove columns from `NUMERIC_COLS` or `CATEGORICAL_COLS`
- Adjust `MODEL_PARAMS` for hyperparameter tuning
- Update file paths if dataset location changes
