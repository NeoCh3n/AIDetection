#!/usr/bin/env python3
"""
NaiveBayesModel - Naive Bayes model implementation.

Python 3.6.8 Compatible
"""

import sys
import os
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.naive_bayes import GaussianNB
from .base import ModelBase


class NaiveBayesModel(ModelBase):
    """Naive Bayes model implementation."""
    
    def create_model(self) -> GaussianNB:
        """Create Naive Bayes model instance."""
        params = self.get_model_params()
        return GaussianNB(**params)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get Naive Bayes hyperparameters."""
        nb_config = self.config.get('model', {}).get('naive_bayes', {})
        return {
            'var_smoothing': nb_config.get('var_smoothing', 1e-9)
        }
    
    def needs_scaling(self) -> bool:
        """Naive Bayes benefits from feature scaling."""
        return True
    
    def get_grid_search_params(self) -> Dict[str, Any]:
        """Get Naive Bayes grid search parameters."""
        return {
            'var_smoothing': [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
        }