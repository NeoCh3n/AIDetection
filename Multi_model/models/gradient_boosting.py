#!/usr/bin/env python3
"""
GradientBoostingModel - Gradient Boosting model implementation.

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

try:
    from sklearn.ensemble import GradientBoostingClassifier
    GRADIENT_BOOSTING_AVAILABLE = True
except ImportError:
    GRADIENT_BOOSTING_AVAILABLE = False

from .base import ModelBase


class GradientBoostingModel(ModelBase):
    """Gradient Boosting model implementation."""
    
    def create_model(self):
        """Create Gradient Boosting model instance."""
        if not GRADIENT_BOOSTING_AVAILABLE:
            raise ImportError("Gradient Boosting not available - sklearn version may be too old")
        
        params = self.get_model_params()
        return GradientBoostingClassifier(**params)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get Gradient Boosting hyperparameters."""
        gb_config = self.config.get('model', {}).get('gradient_boosting', {})
        return {
            'n_estimators': gb_config.get('n_estimators', 100),
            'learning_rate': gb_config.get('learning_rate', 0.1),
            'max_depth': gb_config.get('max_depth', 3),
            'min_samples_split': gb_config.get('min_samples_split', 2),
            'min_samples_leaf': gb_config.get('min_samples_leaf', 1),
            'random_state': gb_config.get('random_state', 42)
        }
    
    def needs_scaling(self) -> bool:
        """Gradient Boosting doesn't require feature scaling."""
        return False
    
    def get_grid_search_params(self) -> Dict[str, Any]:
        """Get Gradient Boosting grid search parameters."""
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        return self.model.feature_importances_