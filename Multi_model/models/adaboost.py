#!/usr/bin/env python3
"""
AdaBoostModel - AdaBoost model implementation.

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

from sklearn.ensemble import AdaBoostClassifier
from .base import ModelBase


class AdaBoostModel(ModelBase):
    """AdaBoost model implementation."""
    
    def create_model(self) -> AdaBoostClassifier:
        """Create AdaBoost model instance."""
        params = self.get_model_params()
        return AdaBoostClassifier(**params)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get AdaBoost hyperparameters."""
        ada_config = self.config.get('model', {}).get('adaboost', {})
        return {
            'n_estimators': ada_config.get('n_estimators', 50),
            'learning_rate': ada_config.get('learning_rate', 1.0),
            'algorithm': ada_config.get('algorithm', 'SAMME.R'),
            'random_state': ada_config.get('random_state', 42)
        }
    
    def needs_scaling(self) -> bool:
        """AdaBoost doesn't require feature scaling."""
        return False
    
    def get_grid_search_params(self) -> Dict[str, Any]:
        """Get AdaBoost grid search parameters."""
        return {
            'n_estimators': [25, 50, 100],
            'learning_rate': [0.5, 1.0, 1.5],
            'algorithm': ['SAMME', 'SAMME.R']
        }
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        return self.model.feature_importances_