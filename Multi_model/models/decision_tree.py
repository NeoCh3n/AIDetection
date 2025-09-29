#!/usr/bin/env python3
"""
DecisionTreeModel - Decision Tree model implementation.

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

from sklearn.tree import DecisionTreeClassifier
from .base import ModelBase


class DecisionTreeModel(ModelBase):
    """Decision Tree model implementation."""
    
    def create_model(self) -> DecisionTreeClassifier:
        """Create Decision Tree model instance."""
        params = self.get_model_params()
        return DecisionTreeClassifier(**params)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get Decision Tree hyperparameters."""
        dt_config = self.config.get('model', {}).get('decision_tree', {})
        return {
            'criterion': dt_config.get('criterion', 'gini'),
            'splitter': dt_config.get('splitter', 'best'),
            'max_depth': dt_config.get('max_depth'),
            'min_samples_split': dt_config.get('min_samples_split', 2),
            'min_samples_leaf': dt_config.get('min_samples_leaf', 1),
            'max_features': dt_config.get('max_features'),
            'class_weight': dt_config.get('class_weight', 'balanced'),
            'random_state': dt_config.get('random_state', 42)
        }
    
    def needs_scaling(self) -> bool:
        """Decision Tree doesn't require feature scaling."""
        return False
    
    def get_grid_search_params(self) -> Dict[str, Any]:
        """Get Decision Tree grid search parameters."""
        return {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
        }
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        return self.model.feature_importances_