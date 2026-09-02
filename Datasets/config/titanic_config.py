"""Configuration for Titanic Dataset"""

# Dataset paths
DATASET_PATH = r"Datasets/titanic/train.csv"
TEST_DATASET_PATH = r"Datasets/titanic/test.csv"
MODEL_PATH = r"models/titanic_model.pkl"

# Target and splitting
TARGET_COLUMN = "Survived"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Columns with numeric and categorical data
NUMERIC_COLS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_COLS = ["Sex", "Embarked"]

# Columns to drop (not useful for prediction)
DROP_COLS = ["PassengerId", "Name", "Ticket", "Cabin"]

# Model parameters
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": RANDOM_STATE
}
