"""
Data Preprocessing Module

Handles all data loading, cleaning, transformation, and preparation
for machine learning models. This module provides reusable functions
that work consistently across all datasets.

Key Functions:
    - load_dataset: Load CSV/JSON files
    - handle_missing_numeric: Fill missing values in numeric columns
    - handle_missing_categorical: Fill missing values in categorical columns
    - remove_duplicates: Remove duplicate rows
    - encode_categorical: Encode categorical variables
    - split_data: Train/test split
    - explore_data: Dataset exploration summary

Example:
    from src.preprocess import load_dataset, handle_missing_numeric
    
    df = load_dataset("path/to/data.csv")
    df = handle_missing_numeric(df, ['Age', 'Score'], strategy='mean')
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional, Union, List


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load dataset from CSV or JSON file.
    
    Parameters
    ----------
    filepath : str
        Path to the data file (CSV or JSON)
    
    Returns
    -------
    pd.DataFrame
        Loaded dataset
    
    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    ValueError
        If file format is not supported
    
    Examples
    --------
    >>> df = load_dataset('Datasets/data.csv')
    >>> print(df.shape)
    (1000, 15)
    """
    try:
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            return pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file format. Use .csv or .json")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at: {filepath}")


def handle_missing_numeric(
    df: pd.DataFrame, 
    columns: List[str], 
    strategy: str = 'mean'
) -> pd.DataFrame:
    """
    Fill missing values in numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        Column names to fill
    strategy : str, default='mean'
        Strategy to use: 'mean', 'median', or 'drop'
    
    Returns
    -------
    pd.DataFrame
        Dataframe with filled missing values
    
    Raises
    ------
    ValueError
        If strategy is not recognized
    
    Examples
    --------
    >>> df = handle_missing_numeric(df, ['Age', 'Score'], strategy='mean')
    >>> print(df.isnull().sum())  # Should show 0 for these columns
    """
    df = df.copy()
    
    if strategy not in ['mean', 'median', 'drop']:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'mean', 'median', or 'drop'")
    
    for col in columns:
        if col not in df.columns:
            print(f"⚠️ Warning: Column '{col}' not found in dataframe")
            continue
        
        if strategy == 'mean':
            fill_value = df[col].mean()
        elif strategy == 'median':
            fill_value = df[col].median()
        elif strategy == 'drop':
            df.dropna(subset=[col], inplace=True)
            continue
        
        df[col].fillna(fill_value, inplace=True)
    
    return df


def handle_missing_categorical(
    df: pd.DataFrame, 
    columns: List[str], 
    strategy: str = 'mode'
) -> pd.DataFrame:
    """
    Fill missing values in categorical columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        Column names to fill
    strategy : str, default='mode'
        Strategy to use: 'mode' or 'drop'
    
    Returns
    -------
    pd.DataFrame
        Dataframe with filled missing values
    
    Raises
    ------
    ValueError
        If strategy is not recognized
    
    Examples
    --------
    >>> df = handle_missing_categorical(df, ['Gender', 'City'], strategy='mode')
    """
    df = df.copy()
    
    if strategy not in ['mode', 'drop']:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'mode' or 'drop'")
    
    for col in columns:
        if col not in df.columns:
            print(f"⚠️ Warning: Column '{col}' not found in dataframe")
            continue
        
        if strategy == 'mode':
            mode_value = df[col].mode()
            if len(mode_value) > 0:
                df[col].fillna(mode_value[0], inplace=True)
        elif strategy == 'drop':
            df.dropna(subset=[col], inplace=True)
    
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    
    Returns
    -------
    pd.DataFrame
        Dataframe without duplicates with reset index
    
    Examples
    --------
    >>> df = remove_duplicates(df)
    >>> print(f"Duplicates removed: {initial_rows - len(df)}")
    """
    initial_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = initial_rows - len(df)
    
    if removed > 0:
        print(f"✓ Removed {removed} duplicate rows")
    
    return df


def encode_categorical(
    df: pd.DataFrame, 
    columns: List[str],
    return_encoders: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Encode categorical columns using LabelEncoder.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        Categorical column names to encode
    return_encoders : bool, default=False
        If True, return the encoders dictionary for later use
    
    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, Dict]
        Encoded dataframe, and optionally the encoders dictionary
    
    Examples
    --------
    >>> X_encoded, encoders = encode_categorical(X, ['Gender', 'City'], return_encoders=True)
    >>> # Save encoders for later use on new data
    """
    df = df.copy()
    encoders = {}
    
    for col in columns:
        if col not in df.columns:
            print(f"⚠️ Warning: Column '{col}' not found in dataframe")
            continue
        
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    if return_encoders:
        return df, encoders
    return df


def scale_features(
    X: Union[pd.DataFrame, np.ndarray],
    fit: bool = True,
    scaler: Optional[StandardScaler] = None
) -> Tuple[Union[pd.DataFrame, np.ndarray], StandardScaler]:
    """
    Scale numerical features using StandardScaler.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature data
    fit : bool, default=True
        If True, fit new scaler; if False, use provided scaler
    scaler : StandardScaler, optional
        Pre-fitted scaler to use if fit=False
    
    Returns
    -------
    Tuple of scaled data and the scaler object
    
    Examples
    --------
    >>> X_train_scaled, scaler = scale_features(X_train, fit=True)
    >>> X_test_scaled, _ = scale_features(X_test, fit=False, scaler=scaler)
    """
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit=False")
        X_scaled = scaler.transform(X)
    
    return X_scaled, scaler


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    test_size : float, default=0.2
        Proportion of test set (0.0 to 1.0)
    random_state : int, default=42
        Random seed for reproducibility
    stratify : bool, default=True
        If True, use stratified split (for classification)
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test)
    
    Examples
    --------
    >>> X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    """
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    print(f"✓ Data split: Train {X_train.shape[0]}, Test {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def explore_data(df: pd.DataFrame, show_sample: bool = True) -> Dict[str, Any]:
    """
    Generate comprehensive dataset exploration summary.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to explore
    show_sample : bool, default=True
        If True, display first few rows
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing exploration results
    
    Examples
    --------
    >>> info = explore_data(df)
    >>> print(f"Dataset size: {info['shape']}")
    """
    print("\n" + "="*70)
    print("📊 DATASET EXPLORATION")
    print("="*70)
    
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    print(f"\n📌 Shape: {info['shape'][0]} rows × {info['shape'][1]} columns")
    print(f"📝 Columns: {', '.join(info['columns'])}")
    
    print(f"\n🔍 Missing Values:")
    missing = info['missing_values']
    if sum(missing.values()) == 0:
        print("   ✓ None")
    else:
        for col, count in missing.items():
            if count > 0:
                print(f"   - {col}: {count}")
    
    print(f"\n🔁 Duplicates: {info['duplicates']}")
    print(f"💾 Memory Usage: {info['memory_usage_mb']:.2f} MB")
    
    if show_sample:
        print(f"\n🔎 First 5 Rows:")
        print(df.head())
    
    print("="*70 + "\n")
    return info
