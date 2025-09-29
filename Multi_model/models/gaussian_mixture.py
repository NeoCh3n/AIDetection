#!/usr/bin/env python3
"""
GaussianMixtureModel - Gaussian Mixture clustering model implementation.

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

from sklearn.mixture import GaussianMixture
from .clustering_base import ClusteringBase


class GaussianMixtureModel(ClusteringBase):
    """Gaussian Mixture Model clustering implementation."""
    
    def create_model(self) -> GaussianMixture:
        """Create Gaussian Mixture model instance."""
        params = self.get_model_params()
        return GaussianMixture(**params)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get Gaussian Mixture hyperparameters."""
        gmm_config = self.config.get('model', {}).get('gaussian_mixture', {})
        return {
            'n_components': gmm_config.get('n_components', 1),
            'covariance_type': gmm_config.get('covariance_type', 'full'),
            'tol': gmm_config.get('tol', 1e-3),
            'reg_covar': gmm_config.get('reg_covar', 1e-6),
            'max_iter': gmm_config.get('max_iter', 100),
            'n_init': gmm_config.get('n_init', 1),
            'init_params': gmm_config.get('init_params', 'kmeans'),
            'random_state': gmm_config.get('random_state', 42)
        }
    
    def needs_scaling(self) -> bool:
        """Gaussian Mixture benefits from feature scaling."""
        return True
    
    def get_grid_search_params(self) -> Dict[str, Any]:
        """Get Gaussian Mixture grid search parameters."""
        return {
            'n_components': [1, 2, 3, 4, 5, 6, 8, 10],
            'covariance_type': ['full', 'tied', 'diag', 'spherical'],
            'max_iter': [100, 200, 300],
            'init_params': ['kmeans', 'random']
        }
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict component probabilities for provided data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Component probabilities
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Apply scaling if used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict_proba(X_scaled)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log-likelihood of samples.
        
        Args:
            X: Feature matrix
            
        Returns:
            Log-likelihood of samples
        """
        if self.model is None:
            raise ValueError("Model must be fitted before scoring")
        
        # Apply scaling if used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.score_samples(X_scaled)
    
    def get_means(self) -> np.ndarray:
        """Get component means."""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        return self.model.means_
    
    def get_covariances(self) -> np.ndarray:
        """Get component covariances."""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        return self.model.covariances_
    
    def get_weights(self) -> np.ndarray:
        """Get component weights."""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        return self.model.weights_
    
    def get_aic(self, X: np.ndarray) -> float:
        """
        Get Akaike Information Criterion.
        
        Args:
            X: Training data used for AIC calculation
            
        Returns:
            AIC score
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        # Apply scaling if used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
            
        return self.model.aic(X_scaled)
    
    def get_bic(self, X: np.ndarray) -> float:
        """
        Get Bayesian Information Criterion.
        
        Args:
            X: Training data used for BIC calculation
            
        Returns:
            BIC score
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        # Apply scaling if used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
            
        return self.model.bic(X_scaled)