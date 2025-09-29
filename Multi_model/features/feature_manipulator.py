#!/usr/bin/env python3
"""
FeatureManipulator - Feature engineering and preprocessing handler.

This module handles time-window aggregation, feature generation, and preprocessing
with consistent interface for both training and detection modes.

Python 3.6.8 Compatible
"""

import sys
import os
import logging
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared_utils.qradar_rule_manager import QRadarRuleManager

# Configure logging
logger = logging.getLogger(__name__)


class FeatureManipulator:
    """
    Feature engineering and preprocessing handler.
    
    Handles time-window aggregation, feature generation, and preprocessing
    with consistent interface for both training and detection modes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FeatureManipulator with configuration.
        
        Args:
            config: Configuration dictionary containing feature engineering settings
        """
        self.config = config
        self.window_size_minutes = config.get('feature_engineering', {}).get('window_size_minutes', 30)
        self.log_transform = config.get('feature_engineering', {}).get('log_transform', True)
        self.rule_manager = QRadarRuleManager(config.get('rule_manager', {}))
        self.logger = logging.getLogger(f"{__name__}.FeatureManipulator")
    
    def process_features(self, df: pd.DataFrame, mode: str = 'train') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process raw data into model-ready features.
        
        Args:
            df: Raw data DataFrame
            mode: 'train' or 'detect'
            
        Returns:
            Tuple of (X, y) where X is feature matrix and y is labels (None for detect mode)
        """
        self.logger.info("Processing features...")
        
        # Aggregate to time windows
        df_agg = self._aggregate_to_windows(df)
        
        # Generate feature vectors
        X, y = self._generate_feature_vectors(df_agg, mode)
        
        self.logger.info(f"Generated feature matrix: {X.shape}")
        
        return X, y
    
    def _aggregate_to_windows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data to time windows."""
        from pipeline.feature_aggregator import aggregate_to_windows
        return aggregate_to_windows(df, window_size_minutes=self.window_size_minutes)
    
    def _generate_feature_vectors(self, df_agg: pd.DataFrame, mode: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate feature vectors from aggregated data."""
        from pipeline.feature_generator import FeatureGenerator
        
        feature_gen = FeatureGenerator(self.rule_manager)
        return feature_gen.generate_feature_vectors(df_agg, mode)