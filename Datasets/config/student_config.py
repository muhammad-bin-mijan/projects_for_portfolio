"""Configuration for Student Performance Dataset"""

# Dataset paths
DATASET_PATH = r"Datasets/Student_Performance/Students Performance Dataset.csv"
MODEL_PATH = r"models/student_model.pkl"

# Target and splitting
TARGET_COLUMN = "Grade"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Columns with numeric and categorical data
NUMERIC_COLS = [
    "Age", "Attendance (%)", "Midterm_Score", "Final_Score",
    "Assignments_Avg", "Quizzes_Avg", "Participation_Score",
    "Projects_Score", "Total_Score", "Study_Hours_per_Week",
    "Stress_Level (1-10)", "Sleep_Hours_per_Night"
]

CATEGORICAL_COLS = [
    "Gender", "Department", "Extracurricular_Activities",
    "Internet_Access_at_Home", "Parent_Education_Level",
    "Family_Income_Level"
]

# Columns to drop (not useful for prediction)
DROP_COLS = ["Student_ID"]

# Model parameters
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "random_state": RANDOM_STATE
}
