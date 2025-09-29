#!/usr/bin/env python3
"""
ExtraTreesModel - Extra Trees model implementation.

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

from sklearn.ensemble import ExtraTreesClassifier
from .base import ModelBase


class ExtraTreesModel(ModelBase):
    """Extra Trees model implementation."""
    
    def create_model(self) -> ExtraTreesClassifier:
        """Create ExtraTrees model instance."""
        params = self.get_model_params()
        return ExtraTreesClassifier(**params)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get ExtraTrees hyperparameters."""
        et_config = self.config.get('model', {}).get('extra_trees', {})
        return {
            'n_estimators': et_config.get('n_estimators', 100),
            'criterion': et_config.get('criterion', 'gini'),
            'max_depth': et_config.get('max_depth'),
            'min_samples_split': et_config.get('min_samples_split', 2),
            'min_samples_leaf': et_config.get('min_samples_leaf', 1),
            'max_features': et_config.get('max_features', 'sqrt'),
            'bootstrap': et_config.get('bootstrap', False),
            'class_weight': et_config.get('class_weight', 'balanced'),
            'random_state': et_config.get('random_state', 42),
            'n_jobs': et_config.get('n_jobs', -1)
        }
    
    def needs_scaling(self) -> bool:
        """ExtraTrees doesn't require feature scaling."""
        return False
    
    def get_grid_search_params(self) -> Dict[str, Any]:
        """Get ExtraTrees grid search parameters."""
        return {
            'n_estimators': [50, 100, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        return self.model.feature_importances_