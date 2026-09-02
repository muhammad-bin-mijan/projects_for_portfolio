"""
Model Training Module

Handles model training, evaluation, and saving. Supports multiple
algorithms for both classification and regression tasks.

Key Functions:
    - train_model: Train a classification/regression model
    - evaluate_model: Evaluate model performance with metrics
    - save_model: Save trained model to disk
    - load_model: Load trained model from disk
    - compare_models: Train and compare multiple models
    - get_feature_importance: Extract feature importance

Example:
    from src.train import train_model, evaluate_model, save_model
    
    model = train_model(X_train, y_train, algorithm='random_forest')
    metrics = evaluate_model(model, X_test, y_test)
    save_model(model, 'models/my_model.pkl')
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, r2_score, mean_absolute_error
)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    algorithm: str = 'random_forest',
    model_type: str = 'classification',
    params: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Train a machine learning model.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target variable
    algorithm : str, default='random_forest'
        Algorithm to use:
        - Classification: 'logistic_regression', 'random_forest', 'gradient_boosting', 'decision_tree', 'knn'
        - Regression: 'linear_regression', 'random_forest', 'gradient_boosting'
    model_type : str, default='classification'
        'classification' or 'regression'
    params : Dict, optional
        Model hyperparameters
    
    Returns
    -------
    Trained model object
    
    Raises
    ------
    ValueError
        If algorithm is not recognized
    
    Examples
    --------
    >>> model = train_model(X_train, y_train, algorithm='random_forest', 
    ...                     params={'n_estimators': 100, 'max_depth': 10})
    """
    if params is None:
        params = {}
    
    models = {
        'classification': {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000, **params),
            'random_forest': RandomForestClassifier(random_state=42, **params),
            'gradient_boosting': GradientBoostingClassifier(random_state=42, **params),
            'decision_tree': DecisionTreeClassifier(random_state=42, **params),
            'knn': KNeighborsClassifier(**params),
        },
        'regression': {
            'linear_regression': LinearRegression(**params),
            'random_forest': RandomForestRegressor(random_state=42, **params),
            'gradient_boosting': GradientBoostingRegressor(random_state=42, **params),
            'decision_tree': DecisionTreeRegressor(random_state=42, **params),
        }
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'classification' or 'regression'")
    
    if algorithm not in models[model_type]:
        available = list(models[model_type].keys())
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {available}")
    
    model = models[model_type][algorithm]
    print(f"🚀 Training {algorithm} ({model_type})...")
    model.fit(X_train, y_train)
    print(f"✓ Model trained successfully")
    
    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = 'classification'
) -> Dict[str, float]:
    """
    Evaluate model performance on test set.
    
    Parameters
    ----------
    model : Any
        Trained model object
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target variable
    model_type : str, default='classification'
        'classification' or 'regression'
    
    Returns
    -------
    Dict[str, float]
        Dictionary of evaluation metrics
    
    Examples
    --------
    >>> metrics = evaluate_model(model, X_test, y_test, model_type='classification')
    >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """
    y_pred = model.predict(X_test)
    
    print("\n" + "="*70)
    print(f"📊 MODEL EVALUATION")
    print("="*70)
    
    if model_type == 'classification':
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        
        print(f"\n✓ Accuracy:  {metrics['accuracy']:.4f}")
        print(f"✓ Precision: {metrics['precision']:.4f}")
        print(f"✓ Recall:    {metrics['recall']:.4f}")
        print(f"✓ F1-Score:  {metrics['f1']:.4f}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))
        
    else:  # regression
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
        }
        
        print(f"\n✓ MSE:  {metrics['mse']:.4f}")
        print(f"✓ RMSE: {metrics['rmse']:.4f}")
        print(f"✓ MAE:  {metrics['mae']:.4f}")
        print(f"✓ R²:   {metrics['r2']:.4f}")
    
    print("="*70 + "\n")
    return metrics


def save_model(model: Any, filepath: str) -> None:
    """
    Save trained model to disk using joblib.
    
    Parameters
    ----------
    model : Any
        Trained model object
    filepath : str
        Path to save the model
    
    Raises
    ------
    Exception
        If save operation fails
    
    Examples
    --------
    >>> save_model(model, 'models/my_model.pkl')
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"✓ Model saved to: {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load trained model from disk.
    
    Parameters
    ----------
    filepath : str
        Path to the saved model
    
    Returns
    -------
    Trained model object
    
    Raises
    ------
    FileNotFoundError
        If model file doesn't exist
    
    Examples
    --------
    >>> model = load_model('models/my_model.pkl')
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Model not found: {filepath}")
    
    model = joblib.load(filepath)
    print(f"✓ Model loaded from: {filepath}")
    return model


def get_feature_importance(
    model: Any,
    feature_names: list,
    top_n: int = 10
) -> Dict[str, float]:
    """
    Extract feature importance from trained model.
    
    Parameters
    ----------
    model : Any
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature column names
    top_n : int, default=10
        Number of top features to return
    
    Returns
    -------
    Dict[str, float]
        Top N features with their importance scores
    
    Raises
    ------
    ValueError
        If model doesn't support feature importance
    
    Examples
    --------
    >>> importance = get_feature_importance(model, X_train.columns, top_n=10)
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model doesn't support feature importance extraction")
    
    importances = model.feature_importances_
    
    if len(feature_names) != len(importances):
        raise ValueError(f"Feature names ({len(feature_names)}) != importances ({len(importances)})")
    
    importance_dict = dict(zip(feature_names, importances))
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🔥 Top {top_n} Important Features:")
    for i, (feature, importance) in enumerate(sorted_importance[:top_n], 1):
        print(f"   {i}. {feature:30s} → {importance:.4f}")
    
    return dict(sorted_importance[:top_n])


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    algorithms: Optional[list] = None,
    model_type: str = 'classification'
) -> Dict[str, Dict]:
    """
    Train and compare multiple models.
    
    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
        Training data
    X_test, y_test : pd.DataFrame, pd.Series
        Test data
    algorithms : list, optional
        List of algorithms to train
    model_type : str, default='classification'
        'classification' or 'regression'
    
    Returns
    -------
    Dict[str, Dict]
        Results for each algorithm
    
    Examples
    --------
    >>> results = compare_models(X_train, y_train, X_test, y_test,
    ...                          algorithms=['random_forest', 'gradient_boosting'])
    """
    if algorithms is None:
        algorithms = ['random_forest', 'decision_tree', 'gradient_boosting']
    
    results = {}
    
    print("\n" + "="*70)
    print(f"🏆 COMPARING {len(algorithms)} MODELS")
    print("="*70 + "\n")
    
    for algorithm in algorithms:
        print(f"\n{'─'*70}")
        model = train_model(X_train, y_train, algorithm=algorithm, model_type=model_type)
        metrics = evaluate_model(model, X_test, y_test, model_type=model_type)
        results[algorithm] = {
            'model': model,
            'metrics': metrics
        }
    
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    # Print summary table
    metrics_names = list(results[algorithms[0]]['metrics'].keys())
    print(f"\n{'Algorithm':<25} {' | '.join(f'{m:>10}' for m in metrics_names)}")
    print("─" * (25 + (len(metrics_names) * 13)))
    
    for algo in algorithms:
        metrics = results[algo]['metrics']
        print(f"{algo:<25} {' | '.join(f'{v:>10.4f}' for v in metrics.values())}")
    
    print("="*70 + "\n")
    
    return results
