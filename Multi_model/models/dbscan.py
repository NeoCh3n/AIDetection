#!/usr/bin/env python3
"""
DBSCANModel - DBSCAN clustering model implementation.

Python 3.6.8 Compatible
"""

import sys
import os
from typing import Dict, Any, Optional
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.cluster import DBSCAN
from .clustering_base import ClusteringBase


class DBSCANModel(ClusteringBase):
    """DBSCAN clustering model implementation."""
    
    def create_model(self) -> DBSCAN:
        """Create DBSCAN model instance."""
        params = self.get_model_params()
        return DBSCAN(**params)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get DBSCAN hyperparameters."""
        dbscan_config = self.config.get('model', {}).get('dbscan', {})
        return {
            'eps': dbscan_config.get('eps', 0.5),
            'min_samples': dbscan_config.get('min_samples', 5),
            'metric': dbscan_config.get('metric', 'euclidean'),
            'algorithm': dbscan_config.get('algorithm', 'auto'),
            'leaf_size': dbscan_config.get('leaf_size', 30),
            'p': dbscan_config.get('p', None),
            'n_jobs': dbscan_config.get('n_jobs', None)
        }
    
    def needs_scaling(self) -> bool:
        """DBSCAN benefits from feature scaling."""
        return True
    
    def get_grid_search_params(self) -> Dict[str, Any]:
        """Get DBSCAN grid search parameters."""
        return {
            'eps': [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
            'min_samples': [3, 5, 7, 10, 15, 20],
            'metric': ['euclidean', 'manhattan', 'cosine']
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        DBSCAN doesn't support prediction on new data.
        This method raises an error with helpful message.
        """
        raise NotImplementedError(
            "DBSCAN doesn't support prediction on new data. "
            "Use fit_predict() to cluster the entire dataset at once."
        )
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """DBSCAN doesn't have cluster centers."""
        return None
    
    def get_core_samples(self) -> Optional[np.ndarray]:
        """Get core sample indices."""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        if hasattr(self.model, 'core_sample_indices_'):
            return self.model.core_sample_indices_
        return None
    
    def get_n_clusters(self) -> int:
        """Get number of clusters found (excluding noise)."""
        if self.cluster_labels_ is None:
            raise ValueError("Model must be fitted first")
        return len(set(self.cluster_labels_)) - (1 if -1 in self.cluster_labels_ else 0)