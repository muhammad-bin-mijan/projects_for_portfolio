# SRC Module - Reusable Code

This folder contains reusable, production-ready Python modules for data science workflows.

## Contents

### `preprocess.py` - Data Preprocessing
Functions for loading, cleaning, and transforming data:
- `load_dataset()` - Load CSV/JSON files
- `handle_missing_numeric()` - Fill missing values in numeric columns
- `handle_missing_categorical()` - Fill missing values in categorical columns
- `remove_duplicates()` - Remove duplicate rows
- `encode_categorical()` - Encode categorical variables
- `scale_features()` - Scale numerical features
- `split_data()` - Train/test split
- `explore_data()` - Dataset exploration summary

### `train.py` - Model Training & Evaluation
Functions for training and evaluating ML models:
- `train_model()` - Train classification/regression models
- `evaluate_model()` - Evaluate with detailed metrics
- `save_model()` - Save trained model to disk
- `load_model()` - Load trained model
- `get_feature_importance()` - Extract feature importance
- `compare_models()` - Train and compare multiple models

### `utils.py` - Utility Functions
Helper functions for visualization and common tasks:
- `print_section()` - Print formatted section headers
- `plot_feature_importance()` - Visualize feature importance
- `plot_confusion_matrix()` - Plot confusion matrix
- `plot_roc_curve()` - Plot ROC curve
- `print_dataset_summary()` - Print dataset statistics
- `get_column_info()` - Get column information

## Usage Examples

```python
# Load and preprocess data
from src.preprocess import load_dataset, handle_missing_numeric, split_data
from Datasets.config.diabetes_config import DATASET_PATH, NUMERIC_COLS_MISSING, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE

df = load_dataset(DATASET_PATH)
df = handle_missing_numeric(df, NUMERIC_COLS_MISSING, strategy='mean')
X_train, X_test, y_train, y_test = split_data(df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN], TEST_SIZE, RANDOM_STATE)

# Train model
from src.train import train_model, evaluate_model, save_model
from Datasets.config.diabetes_config import MODEL_PARAMS, MODEL_PATH

model = train_model(X_train, y_train, algorithm='random_forest', params=MODEL_PARAMS)
metrics = evaluate_model(model, X_test, y_test)
save_model(model, MODEL_PATH)

# Visualize results
from src.utils import plot_feature_importance
plot_feature_importance(model, X_train.columns, top_n=15)
```

## Key Features

✅ **Well-documented** - Every function has detailed docstrings
✅ **Type hints** - Full type annotations for better IDE support
✅ **Error handling** - Proper error messages and validation
✅ **Consistent** - Follows PEP 8 style guide
✅ **Reusable** - Works across all 4 datasets
✅ **Professional** - Production-ready code quality

## See Also

- `Datasets/config/` - Dataset-specific configurations
- `notebooks/` - Analysis notebooks using these modules
- `models/` - Trained model storage
