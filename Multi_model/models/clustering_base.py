#!/usr/bin/env python3
"""
ClusteringBase - Abstract base class for clustering models.

This module defines the interface for unsupervised clustering models
which have different training and prediction patterns than supervised models.

Python 3.6.8 Compatible
"""

import sys
import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import numpy as np
import joblib

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.preprocessing import StandardScaler

# Configure logging
logger = logging.getLogger(__name__)


class ClusteringBase(ABC):
    """
    Abstract base class for clustering models.
    
    Defines the interface that all clustering implementations must follow
    for unsupervised learning workflows.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize clustering model with configuration.
        
        Args:
            config: Model-specific configuration parameters
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.cluster_labels_ = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def create_model(self) -> Any:
        """Create and return the clustering model instance."""
        pass
    
    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """Get model hyperparameters from config."""
        pass
    
    def get_grid_search_params(self) -> Dict[str, Any]:
        """Get grid search parameter grid. Override in subclasses for custom grids."""
        return {}
    
    def fit(self, X: np.ndarray) -> 'ClusteringBase':
        """
        Fit the clustering model on provided data.
        
        Args:
            X: Feature matrix (no labels needed for clustering)
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Fitting clustering model...")
        
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
        
        # Fit model
        self.model.fit(X_scaled)
        
        # Store cluster labels if available
        if hasattr(self.model, 'labels_'):
            self.cluster_labels_ = self.model.labels_
        
        self.logger.info("Clustering model fitting completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for provided data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Cluster labels
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Apply scaling if used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and predict cluster labels in one step.
        
        Args:
            X: Feature matrix
            
        Returns:
            Cluster labels
        """
        self.fit(X)
        
        # For some clustering algorithms, we use the fitted labels
        if self.cluster_labels_ is not None:
            return self.cluster_labels_
        else:
            return self.predict(X)
    
    @abstractmethod
    def needs_scaling(self) -> bool:
        """Return True if this model requires feature scaling."""
        pass
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Get cluster centers if available."""
        if hasattr(self.model, 'cluster_centers_'):
            return self.model.cluster_centers_
        return None
    
    def get_inertia(self) -> Optional[float]:
        """Get inertia (within-cluster sum of squares) if available."""
        if hasattr(self.model, 'inertia_'):
            return self.model.inertia_
        return None
    
    def save_model(self, filepath: str) -> None:
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'model_type': self.__class__.__name__,
            'cluster_labels_': self.cluster_labels_
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Clustering model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'ClusteringBase':
        """Load model from file."""
        model_data = joblib.load(filepath)
        
        # Create instance
        instance = cls(model_data['config'])
        instance.model = model_data['model']
        instance.scaler = model_data.get('scaler')
        instance.cluster_labels_ = model_data.get('cluster_labels_')
        
        return instance