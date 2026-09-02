"""
Source code package for ML pipeline.

This package contains reusable modules for:
- Data preprocessing (preprocess.py)
- Model training and evaluation (train.py)
- Utility functions (utils.py)

Usage:
    from src.preprocess import load_dataset, handle_missing_numeric
    from src.train import train_model, evaluate_model
    from src.utils import print_section, plot_feature_importance
"""

from . import preprocess
from . import train
from . import utils

__version__ = "1.0.0"
__all__ = ['preprocess', 'train', 'utils']
