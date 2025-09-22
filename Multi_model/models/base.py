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
from sklearn.model_selection import GridSearchCV

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
    
    def get_grid_search_params(self) -> Dict[str, Any]:
        """Get grid search parameter grid. Override in subclasses for custom grids."""
        return {}
    
    def train(self, X: np.ndarray, y: np.ndarray, use_grid_search: bool = False) -> 'ModelBase':
        """
        Train the model on provided data.
        
        Args:
            X: Feature matrix
            y: Target labels
            use_grid_search: Whether to use GridSearchCV for hyperparameter tuning
            
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
        
        # Use GridSearchCV if requested and parameters are available
        if use_grid_search:
            param_grid = self.get_grid_search_params()
            if param_grid:
                self.logger.info("Using GridSearchCV for hyperparameter tuning...")
                grid_config = self.config.get('training', {}).get('grid_search', {})
                
                grid_search = GridSearchCV(
                    estimator=self.model,
                    param_grid=param_grid,
                    cv=grid_config.get('cv', 3),
                    scoring=grid_config.get('scoring', 'roc_auc'),
                    verbose=grid_config.get('verbose', 1),
                    n_jobs=grid_config.get('n_jobs', -1)
                )
                
                grid_search.fit(X_scaled, y)
                self.model = grid_search.best_estimator_
                
                self.logger.info(f"Best parameters: {grid_search.best_params_}")
                self.logger.info(f"Best score: {grid_search.best_score_:.4f}")
            else:
                self.logger.warning("GridSearch requested but no parameter grid defined")
                self.model.fit(X_scaled, y)
        else:
            # Train model normally
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