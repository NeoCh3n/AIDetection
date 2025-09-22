#!/usr/bin/env python3
"""
DataHandler - Unified data loading and preprocessing handler.

This module provides a unified interface for loading data from different sources
(CSV files for training, MongoDB for detection) with consistent output format.

Python 3.6.8 Compatible
"""

import sys
import os
import logging
from typing import Dict, Any
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared_utils.time_utils import parse_qradar_timestamp

# Configure logging
logger = logging.getLogger(__name__)


class DataHandler:
    """
    Unified data loading and preprocessing handler.
    
    Abstracts data sources (CSV files for training, MongoDB for detection)
    and provides consistent output format for downstream processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataHandler with configuration.
        
        Args:
            config: Configuration dictionary containing data paths and settings
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DataHandler")
    
    def load_data(self, mode: str) -> pd.DataFrame:
        """
        Load data based on mode with unified output format.
        
        Args:
            mode: Either 'train' or 'detect'
            
        Returns:
            Standardized DataFrame with columns: hostname, rule_id, timestamp, count, source_label
        """
        if mode not in ['train', 'detect']:
            raise ValueError("Mode must be either 'train' or 'detect'")
        
        self.logger.info(f"Loading data in {mode} mode...")
        
        try:
            if mode == 'train':
                df = self._load_training_data()
            else:
                df = self._load_detection_data()
            
            # Validate and preprocess
            df = self._validate_and_preprocess(df)
            
            self.logger.info(f"Successfully loaded {len(df)} records in {mode} mode")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data in {mode} mode: {e}")
            raise
    
    def _load_training_data(self) -> pd.DataFrame:
        """Load training data from CSV files."""
        from pipeline.data_loader import _load_training_data
        return _load_training_data(self.config)
    
    def _load_detection_data(self) -> pd.DataFrame:
        """Load detection data from MongoDB."""
        from pipeline.data_loader import _load_detection_data
        return _load_detection_data(self.config)
    
    def _validate_and_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and preprocess loaded data."""
        required_columns = ['hostname', 'rule_id', 'timestamp', 'count']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure proper data types
        df['rule_id'] = pd.to_numeric(df['rule_id'], errors='coerce')
        df['count'] = pd.to_numeric(df['count'], errors='coerce') 
        df['hostname'] = df['hostname'].astype(str)
        
        # Parse timestamps
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            if df['timestamp'].dtype == 'object':
                df['timestamp'] = df['timestamp'].apply(parse_qradar_timestamp)
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Drop invalid rows
        df = df.dropna(subset=['rule_id', 'count', 'hostname'])
        
        return df