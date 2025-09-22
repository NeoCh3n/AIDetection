#!/usr/bin/env python3
"""
RandomForestModel - Random Forest model implementation.

Python 3.6.8 Compatible
"""

import sys
import os
from typing import Dict, Any
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.ensemble import RandomForestClassifier
from .base import ModelBase


class RandomForestModel(ModelBase):
    """Random Forest model implementation."""
    
    def create_model(self) -> RandomForestClassifier:
        """Create RandomForest model instance."""
        params = self.get_model_params()
        return RandomForestClassifier(**params)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get RandomForest hyperparameters."""
        rf_config = self.config.get('model', {}).get('random_forest', {})
        return {
            'n_estimators': rf_config.get('n_estimators', 200),
            'class_weight': rf_config.get('class_weight', 'balanced_subsample'),
            'max_features': rf_config.get('max_features', 'sqrt'),
            'random_state': rf_config.get('random_state', 42),
            'n_jobs': rf_config.get('n_jobs', -1),
            'max_depth': rf_config.get('max_depth'),
            'min_samples_split': rf_config.get('min_samples_split', 2),
            'min_samples_leaf': rf_config.get('min_samples_leaf', 1)
        }
    
    def needs_scaling(self) -> bool:
        """RandomForest doesn't require feature scaling."""
        return False
    
    def get_grid_search_params(self) -> Dict[str, Any]:
        """Get RandomForest grid search parameters."""
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample']
        }
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        return self.model.feature_importances_