#!/usr/bin/env python3
"""
ModelBase - Abstract base class for machine learning models.

This module defines the interface that all model implementations must follow
to ensure consistent behavior across different algorithms.

Python 3.6.8 Compatible
"""

import sys
import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
import joblib

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.preprocessing import StandardScaler

# Configure logging
logger = logging.getLogger(__name__)


class ModelBase(ABC):
    """
    Abstract base class for machine learning models.
    
    Defines the interface that all model implementations must follow
    to ensure consistent behavior across different algorithms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model with configuration.
        
        Args:
            config: Model-specific configuration parameters
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def create_model(self) -> Any:
        """Create and return the model instance."""
        pass
    
    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """Get model hyperparameters from config."""
        pass
    
    def train(self, X: np.ndarray, y: np.ndarray) -> 'ModelBase':
        """
        Train the model on provided data.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Training model...")
        
        # Create model if not exists
        if self.model is None:
            self.model = self.create_model()
        
        # Apply scaling if needed
        if self.needs_scaling():
            if self.scaler is None:
                self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Train model
        self.model.fit(X_scaled, y)
        
        self.logger.info("Model training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on provided data."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Apply scaling if used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError(f"{self.__class__.__name__} does not support probability predictions")
        
        # Apply scaling if used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict_proba(X_scaled)
    
    @abstractmethod
    def needs_scaling(self) -> bool:
        """Return True if this model requires feature scaling."""
        pass
    
    def save_model(self, filepath: str) -> None:
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'model_type': self.__class__.__name__
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'ModelBase':
        """Load model from file."""
        model_data = joblib.load(filepath)
        
        # Create instance
        instance = cls(model_data['config'])
        instance.model = model_data['model']
        instance.scaler = model_data.get('scaler')
        
        return instance