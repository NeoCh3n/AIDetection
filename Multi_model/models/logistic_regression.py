#!/usr/bin/env python3
"""
LogisticRegressionModel - Logistic Regression model implementation.

Python 3.6.8 Compatible
"""

import sys
import os
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.linear_model import LogisticRegression
from .base import ModelBase


class LogisticRegressionModel(ModelBase):
    """Logistic Regression model implementation."""
    
    def create_model(self) -> LogisticRegression:
        """Create Logistic Regression model instance."""
        params = self.get_model_params()
        return LogisticRegression(**params)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get Logistic Regression hyperparameters."""
        lr_config = self.config.get('model', {}).get('logistic_regression', {})
        return {
            'C': lr_config.get('C', 1.0),
            'penalty': lr_config.get('penalty', 'l2'),
            'solver': lr_config.get('solver', 'liblinear'),
            'max_iter': lr_config.get('max_iter', 1000),
            'random_state': lr_config.get('random_state', 42),
            'class_weight': lr_config.get('class_weight', 'balanced')
        }
    
    def needs_scaling(self) -> bool:
        """Logistic Regression benefits from feature scaling."""
        return True