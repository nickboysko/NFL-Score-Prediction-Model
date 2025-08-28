"""
NFL Score Prediction Model Package

This package contains modules for:
- data_loader: Loading and preprocessing NFL schedule data
- features: Feature engineering for the model
- model: Training and evaluation of the XGBoost model
- infer: Making predictions on upcoming games
"""

from . import data_loader
from . import features
from . import model
from . import infer

__version__ = "0.1.0"
__all__ = ["data_loader", "features", "model", "infer"]
