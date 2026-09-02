"""
Utility Functions Module

Helper functions for data exploration, visualization, and general utilities
used across the project.

Key Functions:
    - print_section: Print formatted section headers
    - plot_feature_importance: Visualize feature importance
    - print_dataset_summary: Print dataset statistics

Example:
    from src.utils import print_section, plot_feature_importance
    
    print_section("Data Loading")
    plot_feature_importance(model, feature_names)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional


def print_section(title: str, width: int = 70) -> None:
    """
    Print a formatted section header.
    
    Parameters
    ----------
    title : str
        Section title to display
    width : int, default=70
        Width of the section line
    
    Examples
    --------
    >>> print_section("Data Loading")
    """
    print("\n" + "="*width)
    print(f"  {title}")
    print("="*width + "\n")


def print_subsection(title: str, width: int = 70) -> None:
    """
    Print a formatted subsection header.
    
    Parameters
    ----------
    title : str
        Subsection title to display
    width : int, default=70
        Width of the section line
    
    Examples
    --------
    >>> print_subsection("Feature Encoding")
    """
    print("\n" + "─"*width)
    print(f"  {title}")
    print("─"*width + "\n")


def plot_feature_importance(
    model: any,
    feature_names: List[str],
    top_n: int = 15,
    figsize: tuple = (12, 6)
) -> None:
    """
    Plot top N important features from a trained model.
    
    Parameters
    ----------
    model : Any
        Trained model with feature_importances_ attribute
    feature_names : List[str]
        List of feature column names
    top_n : int, default=15
        Number of top features to display
    figsize : tuple, default=(12, 6)
        Figure size (width, height)
    
    Raises
    ------
    ValueError
        If model doesn't support feature importance
    
    Examples
    --------
    >>> plot_feature_importance(model, X_train.columns, top_n=15)
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model doesn't support feature importance visualization")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=figsize)
    sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
    plt.title(f'Top {top_n} Features by Importance', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    figsize: tuple = (8, 6)
) -> None:
    """
    Plot confusion matrix for classification tasks.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    title : str, default="Confusion Matrix"
        Title of the plot
    figsize : tuple, default=(8, 6)
        Figure size (width, height)
    
    Examples
    --------
    >>> plot_confusion_matrix(y_test, y_pred, title="Diabetes Model CM")
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str = "ROC Curve",
    figsize: tuple = (8, 6)
) -> None:
    """
    Plot ROC curve for binary classification.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_pred_proba : np.ndarray
        Predicted probabilities for positive class
    title : str, default="ROC Curve"
        Title of the plot
    figsize : tuple, default=(8, 6)
        Figure size (width, height)
    
    Examples
    --------
    >>> plot_roc_curve(y_test, y_pred_proba, title="Diabetes ROC")
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_dataset_summary(df: pd.DataFrame) -> None:
    """
    Print comprehensive dataset summary statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to summarize
    
    Examples
    --------
    >>> print_dataset_summary(df)
    """
    print("\n" + "="*70)
    print("📊 DATASET SUMMARY")
    print("="*70)
    
    print(f"\n📌 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print(f"\n📝 Data Types:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"   - {dtype}: {count} columns")
    
    print(f"\n🔍 Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✓ None")
    else:
        for col, count in missing[missing > 0].items():
            pct = (count / len(df)) * 100
            print(f"   - {col}: {count} ({pct:.1f}%)")
    
    print(f"\n🔁 Duplicates: {df.duplicated().sum()}")
    
    print(f"\n💾 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n" + "="*70 + "\n")


def get_column_info(df: pd.DataFrame) -> dict:
    """
    Get detailed information about dataset columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze
    
    Returns
    -------
    dict
        Dictionary containing column information
    
    Examples
    --------
    >>> info = get_column_info(df)
    >>> print(info['numeric_cols'])
    """
    info = {
        'numeric_cols': df.select_dtypes(include=['number']).columns.tolist(),
        'categorical_cols': df.select_dtypes(include=['object']).columns.tolist(),
        'total_cols': len(df.columns),
        'total_rows': len(df),
    }
    return info
