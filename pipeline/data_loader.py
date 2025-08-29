"""
Data Loader Module - Unified data loading for training and detection modes.

This module provides standardized data loading functionality for both training
and detection modes, abstracting away the differences between CSV files and
MongoDB data sources.

Python 3.6.8 Compatible
"""

import os
import sys
import pandas as pd
from typing import Dict, Any
import logging
import glob
from datetime import datetime, timedelta
import pytz

# Add system path for config access
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import system config
import system.config as config

# Import shared utilities
from shared_utils.time_utils import parse_qradar_timestamp
from mongodb.mongodb_connection import get_mongodb_manager

# Configure unified logging
logger = logging.getLogger(__name__)


def load_data(mode: str, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load data from appropriate source based on mode.
    
    Unified data loading for both training and detection modes with consistent
    output format and proper integration with the MongoDBConnectionManager.
    
    Args:
        mode: Either 'train' or 'detect'
        config: Configuration dictionary containing paths and settings
        
    Returns:
        Standardized DataFrame with columns: hostname, rule_id, timestamp, count, source_label
        
    Raises:
        ValueError: If mode is not 'train' or 'detect'
        FileNotFoundError: If required data files are not found
    """
    if mode not in ['train', 'detect']:
        raise ValueError("Mode must be either 'train' or 'detect'")
    
    logger.info(f"Loading data in {mode} mode...")
    
    try:
        if mode == 'train':
            df = _load_training_data(config)
        else:
            df = _load_detection_data(config)
        
        # Validate unified output format
        if not validate_data(df):
            raise ValueError("Loaded data does not meet unified schema requirements")
        
        logger.info(f"Successfully loaded {len(df)} records in {mode} mode")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load data in {mode} mode: {e}")
        raise


def _load_training_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load training data from CSV files with unified schema.
    
    Args:
        config: Configuration dictionary with training data paths
        
    Returns:
        Standardized DataFrame with columns: hostname, rule_id, timestamp, count, source_label
    """
    logger.info("Loading training data from CSV files...")
    
    # Get configuration paths with glob support
    normal_data_path = config.get('training_data_path', './Training_data/normal')
    attack_data_path = config.get('attack_data_path', './Training_data/attack')
    
    # Use glob to find actual training data folders
    normal_dirs = glob.glob(normal_data_path) or [normal_data_path]
    attack_dirs = glob.glob(attack_data_path) or [attack_data_path]
    
    actual_normal_path = normal_dirs[0] if normal_dirs else normal_data_path
    actual_attack_path = attack_dirs[0] if attack_dirs else attack_data_path
    
    # Load normal activity data
    normal_df = _load_csv_files(actual_normal_path, source_label='normal')
    logger.info(f"Loaded {len(normal_df)} normal activity records")
    
    # Load attack data
    attack_df = _load_csv_files(actual_attack_path, source_label='attack')
    logger.info(f"Loaded {len(attack_df)} attack activity records")
    
    # Combine datasets
    combined_df = pd.concat([normal_df, attack_df], ignore_index=True)
    logger.info(f"Total training records: {len(combined_df)}")
    
    return combined_df


def _load_csv_files(data_path: str, source_label: str) -> pd.DataFrame:
    """
    Load CSV files from specified directory with unified schema preprocessing.
    
    Expected CSV structure:
    - sysmon_hostname (custom) -> hostname
    - Custom Rule -> rule_id  
    - Log Source Time (Minimum) -> timestamp
    - Count -> count
    
    Performs inline preprocessing:
    - Timestamp parsing using parse_qradar_timestamp()
    - Data type conversion (rule_id, count to int; hostname to string)
    - Missing value handling
    
    Args:
        data_path: Path to directory containing CSV files
        source_label: Label to identify data source ('normal' or 'attack')
        
    Returns:
        DataFrame with unified columns: hostname, rule_id, timestamp, count, source_label
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    all_files = []
    
    # Column mapping for QRadar CSV structure
    column_mapping = {
        'sysmon_hostname (custom)': 'hostname',
        'Custom Rule': 'rule_id',
        'Log Source Time (Minimum)': 'timestamp_str',
        'Count': 'count'
    }
    
    # Find all CSV files in directory
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    if not csv_files:
        logger.warning(f"No CSV files found in {data_path}")
        return pd.DataFrame(columns=['hostname', 'rule_id', 'timestamp', 'count', 'source_label'])
    
    for file in csv_files:
        file_path = os.path.join(data_path, file)
        try:
            logger.info(f"Processing {file}...")
            df = pd.read_csv(file_path)
            
            # Map actual columns to standardized names
            df = df.rename(columns=column_mapping)
            
            # Validate required columns
            required_cols = ['hostname', 'rule_id', 'timestamp_str', 'count']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Skipping {file}: missing columns {missing_cols}")
                continue
            
            # Process timestamps
            df['timestamp'] = df['timestamp_str'].apply(
                lambda x: parse_qradar_timestamp(str(x).strip())
            )
            
            # Add source label
            df['source_label'] = source_label
            
            # Ensure correct data types
            df['rule_id'] = pd.to_numeric(df['rule_id'], errors='coerce')
            df['count'] = pd.to_numeric(df['count'], errors='coerce')
            df['hostname'] = df['hostname'].astype(str)
            
            # Convert to nullable integer types
            df['rule_id'] = df['rule_id'].astype('Int64')
            df['count'] = df['count'].astype('Int64')
            
            # Drop rows with invalid data
            df = df.dropna(subset=['hostname', 'rule_id', 'timestamp', 'count'])
            
            # Ensure consistent column order
            df = df[['hostname', 'rule_id', 'timestamp', 'count', 'source_label']]
            
            all_files.append(df)
            
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
            continue
    
    if not all_files:
        logger.warning(f"No valid data loaded from {data_path}")
        return pd.DataFrame(columns=['hostname', 'rule_id', 'timestamp', 'count', 'source_label'])
    
    return pd.concat(all_files, ignore_index=True)


def _load_detection_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load detection data from MongoDB using MongoDBConnectionManager.
    
    Uses the unified MongoDB connection manager to query recent detection windows
    and transforms the data into the standardized format.
    
    Args:
        config: Configuration dictionary with MongoDB settings
        
    Returns:
        Standardized DataFrame with columns: hostname, rule_id, timestamp, count, source_label
    """
    logger.info("Loading detection data from MongoDB using unified connection...")
    
    try:
        # Get configuration parameters
        query_window_minutes = config.get('fetch_data_frequency_default', 15)
        
        # Set timezone
        tz = pytz.UTC
        
        # Calculate time window
        end_time = datetime.now(tz)
        start_time = end_time - timedelta(minutes=query_window_minutes)
        
        # Use MongoDB connection manager
        with get_mongodb_manager() as manager:
            if not manager.connect():
                raise RuntimeError("Failed to connect to MongoDB")
            
            # Get unlabeled windows for detection
            windows = manager.get_unlabeled_windows(start_time, end_time)
            
            if not windows:
                logger.warning(f"No detection data found for window: {start_time} to {end_time}")
                return pd.DataFrame(columns=['hostname', 'rule_id', 'timestamp', 'count', 'source_label'])
            
            # Transform windows to unified format
            rows = []
            for window in windows:
                window_start = window.get('window_start')

                # Prefer host-level breakdown when available (AQL inserter schema)
                host_triggers = window.get('host_triggers') or {}
                if isinstance(host_triggers, dict) and host_triggers:
                    for host, payload in host_triggers.items():
                        rules = (payload or {}).get('rules') or {}
                        for rule_id, count in (rules.items() if isinstance(rules, dict) else []):
                            try:
                                if count is not None and int(count) > 0:
                                    rows.append({
                                        'hostname': str(host),
                                        'rule_id': int(rule_id),
                                        'timestamp': window_start,
                                        'count': int(count),
                                        'source_label': 'detection'
                                    })
                            except Exception:
                                continue
                    continue  # handled this window fully

                # Handle alternate unified shapes: rule_counts/features
                hostname = window.get('hostname', 'global')
                rule_counts = window.get('rule_counts', {})
                if not rule_counts:
                    rule_counts = window.get('features', {})
                if not rule_counts:
                    rule_counts = window.get('feature_vector', {})  # AQL inserter total vector

                for rule_id, count in (rule_counts.items() if isinstance(rule_counts, dict) else []):
                    try:
                        if count is not None and int(count) > 0:  # Only include positive counts
                            rows.append({
                                'hostname': str(hostname),
                                'rule_id': int(rule_id),
                                'timestamp': window_start,
                                'count': int(count),
                                'source_label': 'detection'
                            })
                    except Exception:
                        continue
            
            # Create DataFrame
            if not rows:
                logger.warning("No rule triggers found in detection windows")
                return pd.DataFrame(columns=['hostname', 'rule_id', 'timestamp', 'count', 'source_label'])
            
            df = pd.DataFrame(rows)

            # Ensure timestamp is proper datetime dtype
            if not df.empty:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except Exception:
                    pass

            # Ensure correct data types
            df['rule_id'] = pd.to_numeric(df['rule_id'], errors='coerce')
            df['count'] = pd.to_numeric(df['count'], errors='coerce')
            df['hostname'] = df['hostname'].astype(str)
            
            # Convert to nullable integer types
            df['rule_id'] = df['rule_id'].astype('Int64')
            df['count'] = df['count'].astype('Int64')
            
            # Drop rows with invalid data
            df = df.dropna(subset=['hostname', 'rule_id', 'timestamp', 'count'])
            
            logger.info(f"Loaded {len(df)} detection records from {len(windows)} windows")
            return df
            
    except Exception as e:
        logger.error(f"Error loading detection data from MongoDB: {e}")
        raise


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the loaded data meets unified schema requirements.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if data is valid, False otherwise
    """
    if df.empty:
        logger.warning("Loaded data is empty")
        return True  # Allow empty DataFrames for edge cases
    
    required_columns = ['hostname', 'rule_id', 'timestamp', 'count', 'source_label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check for required data types
    try:
        # Validate rule_id contains integers
        df['rule_id'] = pd.to_numeric(df['rule_id'], errors='coerce')
        if df['rule_id'].isna().any():
            logger.error("Found non-numeric values in rule_id column")
            return False
        
        # Validate count contains non-negative integers
        df['count'] = pd.to_numeric(df['count'], errors='coerce')
        if df['count'].isna().any():
            logger.error("Found non-numeric values in count column")
            return False
        if (df['count'] < 0).any():
            logger.error("Found negative values in count column")
            return False
        
        # Validate hostname contains strings
        if not df['hostname'].apply(lambda x: isinstance(x, str)).all():
            logger.error("Found non-string values in hostname column")
            return False
        
        # Validate timestamp contains datetime objects
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            logger.error("timestamp column is not datetime type")
            return False
        
        # Validate source_label
        valid_labels = {'normal', 'attack', 'detection'}
        unique_labels = set(str(label) for label in df['source_label'].tolist())
        invalid_labels = unique_labels - valid_labels
        if invalid_labels:
            logger.warning(f"Found unexpected source labels: {invalid_labels}")
    
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return False
    
    logger.debug("Data validation passed")
    return True


if __name__ == "__main__":
    """
    Test the data loader with both training and detection modes.
    """
    import logging
    
    # Set up logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("Testing Unified Data Loader...")
    print("=" * 50)
    
    # Test training mode
    print("\n1. Testing Training Mode...")
    train_config = {
        'training_data_path': './Training_data/normal',
        'attack_data_path': './Training_data/attack'
    }
    
    try:
        train_df = load_data('train', train_config)
        print(f"✅ Training data loaded: {len(train_df)} records")
        print(f"   Columns: {list(train_df.columns)}")
        print(f"   Source labels: {train_df['source_label'].value_counts().to_dict()}")
    except Exception as e:
        print(f"❌ Training mode failed: {e}")
    
    # Test detection mode
    print("\n2. Testing Detection Mode...")
    detect_config = {
        'query_frequency_minutes': 30,
        'timezone': 'UTC'
    }
    
    try:
        detect_df = load_data('detect', detect_config)
        print(f"✅ Detection data loaded: {len(detect_df)} records")
        print(f"   Columns: {list(detect_df.columns)}")
        print(f"   Unique hosts: {detect_df['hostname'].nunique()}")
    except Exception as e:
        print(f"❌ Detection mode failed: {e}")
    
    print("\n" + "=" * 50)
    print("Data Loader Test Complete")
