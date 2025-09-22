#!/usr/bin/env python3
"""
SVMModel - Support Vector Machine model implementation.

Python 3.6.8 Compatible
"""

import sys
import os
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.svm import SVC
from .base import ModelBase


class SVMModel(ModelBase):
    """Support Vector Machine model implementation."""
    
    def create_model(self) -> SVC:
        """Create SVM model instance."""
        params = self.get_model_params()
        return SVC(**params)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get SVM hyperparameters."""
        svm_config = self.config.get('model', {}).get('svm', {})
        return {
            'C': svm_config.get('C', 1.0),
            'kernel': svm_config.get('kernel', 'rbf'),
            'gamma': svm_config.get('gamma', 'scale'),
            'probability': svm_config.get('probability', True),
            'random_state': svm_config.get('random_state', 42),
            'class_weight': svm_config.get('class_weight', 'balanced')
        }
    
    def needs_scaling(self) -> bool:
        """SVM requires feature scaling."""
        return True