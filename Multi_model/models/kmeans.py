#!/usr/bin/env python3
"""
KMeansModel - K-Means clustering model implementation.

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

from sklearn.cluster import KMeans
from .clustering_base import ClusteringBase


class KMeansModel(ClusteringBase):
    """K-Means clustering model implementation."""
    
    def create_model(self) -> KMeans:
        """Create K-Means model instance."""
        params = self.get_model_params()
        return KMeans(**params)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get K-Means hyperparameters."""
        kmeans_config = self.config.get('model', {}).get('kmeans', {})
        return {
            'n_clusters': kmeans_config.get('n_clusters', 8),
            'init': kmeans_config.get('init', 'k-means++'),
            'n_init': kmeans_config.get('n_init', 10),
            'max_iter': kmeans_config.get('max_iter', 300),
            'tol': kmeans_config.get('tol', 1e-4),
            'random_state': kmeans_config.get('random_state', 42),
            'algorithm': kmeans_config.get('algorithm', 'auto')
        }
    
    def needs_scaling(self) -> bool:
        """K-Means benefits from feature scaling."""
        return True
    
    def get_grid_search_params(self) -> Dict[str, Any]:
        """Get K-Means grid search parameters."""
        return {
            'n_clusters': [2, 3, 4, 5, 6, 7, 8, 10, 12, 15],
            'init': ['k-means++', 'random'],
            'n_init': [10, 20],
            'max_iter': [300, 500]
        }
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers."""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        return self.model.cluster_centers_
    
    def get_inertia(self) -> float:
        """Get within-cluster sum of squares."""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        return self.model.inertia_