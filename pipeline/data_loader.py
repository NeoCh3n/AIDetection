"""
Data Loader Module - Unified data loading for training and detection modes.

This module provides standardized data loading functionality for both training
and detection modes, abstracting away the differences between CSV files and
MongoDB data sources.
"""

import os
import pandas as pd
from typing import Dict, Any, Optional
import logging
import glob
import json
from datetime import datetime, timedelta
import pytz

try:
    import pymongo
    from pymongo import MongoClient
except ImportError:
    pymongo = None
    MongoClient = None

# Import time utilities for inline timestamp processing
from shared_utils.time_utils import parse_qradar_timestamp, categorize_query_timestamp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(mode: str, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load data from appropriate source based on mode.
    
    Args:
        mode: Either 'train' or 'detect'
        config: Configuration dictionary containing paths and settings
        
    Returns:
        Standardized DataFrame ready for preprocessing
        
    Raises:
        ValueError: If mode is not 'train' or 'detect'
        FileNotFoundError: If required data files are not found
    """
    if mode not in ['train', 'detect']:
        raise ValueError("Mode must be either 'train' or 'detect'")
    
    if mode == 'train':
        return _load_training_data(config)
    else:
        return _load_detection_data(config)


def _load_training_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load training data from CSV files.
    
    Args:
        config: Configuration dictionary with training data paths
        
    Returns:
        Standardized DataFrame with columns: hostname, rule_id, timestamp, count, source_label
    """
    logger.info("Loading training data from CSV files...")
    
    # Get configuration paths with glob support for flexible folder names
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
    Load CSV files from specified directory with actual column mapping and inline preprocessing.
    
    Expected CSV columns:
    - sysmon_hostname (custom) -> hostname
    - Custom Rule -> rule_id  
    - Log Source Time (Minimum) -> timestamp_str (then processed to datetime)
    - Count -> count
    
    Includes inline preprocessing:
    - Timestamp parsing using time_utils.parse_qradar_timestamp()
    - Boundary adjustment using categorize_query_timestamp() with 5-second tolerance
    - Data type conversion (rule_id, count to int; hostname to string)
    
    Args:
        data_path: Path to directory containing CSV files
        source_label: Label to identify data source ('normal' or 'attack')
        
    Returns:
        DataFrame with standardized columns: hostname, rule_id, timestamp, count, source_label
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    all_files = []
    
    # Column mapping for actual CSV structure
    column_mapping = {
        'sysmon_hostname (custom)': 'hostname',
        'Custom Rule': 'rule_id',
        'Log Source Time (Minimum)': 'timestamp_str',
        'Count': 'count'
    }
    
    # Find all CSV files in directory
    for file in os.listdir(data_path):
        if file.endswith('.csv'):
            file_path = os.path.join(data_path, file)
            try:
                df = pd.read_csv(file_path)
                
                # Map actual columns to standardized column names
                df = df.rename(columns=column_mapping)
                
                # Validate required columns after mapping
                required_cols = ['hostname', 'rule_id', 'timestamp_str', 'count']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    logger.warning(f"Missing columns in {file}: {missing_cols}. Available: {list(df.columns)}")
                    continue
                
                # Process timestamps inline using time_utils
                logger.info(f"Processing timestamps in {file}...")
                df['timestamp'] = df['timestamp_str'].apply(
                    lambda x: categorize_query_timestamp(parse_qradar_timestamp(str(x).strip()))
                )
                
                # Add source label
                df['source_label'] = source_label
                
                # Ensure correct data types
                df['rule_id'] = pd.to_numeric(df['rule_id'], errors='coerce')
                df['count'] = pd.to_numeric(df['count'], errors='coerce')
                df['hostname'] = df['hostname'].astype(str)
                
                # Convert to nullable integer types, handling potential NaN values
                df['rule_id'] = df['rule_id'].astype('Int64')
                df['count'] = df['count'].astype('Int64')
                
                # Drop rows with invalid data
                df = df.dropna(subset=['hostname', 'rule_id', 'timestamp', 'count'])
                
                # Keep only standard columns
                df = df[['hostname', 'rule_id', 'timestamp', 'count', 'source_label']]
                
                all_files.append(df)
                
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
                continue
    
    if not all_files:
        logger.warning(f"No valid CSV files found in {data_path}")
        return pd.DataFrame(columns=['hostname', 'rule_id', 'timestamp', 'count', 'source_label'])
    
    return pd.concat(all_files, ignore_index=True)


def _load_detection_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load detection data from MongoDB JSON format.
    
    Args:
        config: Configuration dictionary with MongoDB settings
        
    Returns:
        Standardized DataFrame with columns: hostname, rule_id, timestamp, count, source_label
    """
    if pymongo is None:
        raise ImportError("pymongo is required for detection mode. Install with: pip install pymongo")
    
    logger.info("Loading detection data from MongoDB...")
    
    # Load MongoDB configuration
    mongodb_config_path = config.get('mongodb_config', './mongodb/mongodb_config.json')
    try:
        with open(mongodb_config_path, 'r') as f:
            mongodb_config = json.load(f)
    except FileNotFoundError:
        logger.error(f"MongoDB config file not found: {mongodb_config_path}")
        raise
    
    # Extract MongoDB settings
    mongo_config = mongodb_config['mongodb']
    collection_name = mongodb_config['collections']['detection_windows']
    
    # Calculate query window based on configuration
    query_window_minutes = mongodb_config['pipeline']['query_frequency_minutes']
    timezone = mongodb_config['pipeline']['timezone']
    
    # Set timezone
    tz = pytz.timezone('Asia/Hong_Kong') if timezone == 'HKT' else pytz.UTC
    
    # Calculate time window
    end_time = datetime.now(tz)
    start_time = end_time - timedelta(minutes=query_window_minutes)
    
    try:
        # Connect to MongoDB
        if MongoClient is None:
            raise ImportError("MongoClient not available - pymongo not installed")
        
        client = MongoClient(mongo_config['connection_string'])
        db = client[mongo_config['db_name']]
        collection = db[collection_name]
        
        # Query recent detection windows
        query = {
            'window_start': {
                '$gte': start_time,
                '$lte': end_time
            }
        }
        
        cursor = collection.find(query)
        documents = list(cursor)
        
        if not documents:
            logger.warning(f"No detection data found for window: {start_time} to {end_time}")
            return pd.DataFrame(columns=['hostname', 'rule_id', 'timestamp', 'count', 'source_label'])
        
        # Transform MongoDB JSON to DataFrame
        rows = []
        for doc in documents:
            window_start = doc.get('window_start')
            rule_counts = doc.get('rule_counts', {})
            host_triggers = doc.get('host_triggers', {})
            
            # Extract hostname from host_triggers
            hostname = 'unknown'
            if host_triggers:
                # Get first hostname from host_triggers keys
                hostname = next(iter(host_triggers.keys()), 'unknown')
            
            # Flatten rule_counts into individual rows
            for rule_id, count in rule_counts.items():
                rows.append({
                    'hostname': hostname,
                    'rule_id': int(rule_id),
                    'timestamp': window_start,
                    'count': int(count),
                    'source_label': 'detection'
                })
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Ensure correct data types
        df['rule_id'] = pd.to_numeric(df['rule_id'], errors='coerce')
        df['count'] = pd.to_numeric(df['count'], errors='coerce')
        df['hostname'] = df['hostname'].astype(str)
        
        # Convert to nullable integer types
        df['rule_id'] = df['rule_id'].astype('Int64')
        df['count'] = df['count'].astype('Int64')
        
        # Drop rows with invalid data
        df = df.dropna(subset=['hostname', 'rule_id', 'timestamp', 'count'])
        
        logger.info(f"Loaded {len(df)} detection records from MongoDB")
        return df
        
    except Exception as e:
        logger.error(f"Error loading detection data from MongoDB: {e}")
        raise
    finally:
        if 'client' in locals():
            client.close()


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the loaded data meets requirements.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if data is valid, False otherwise
    """
    if df.empty:
        logger.error("Loaded data is empty")
        return False
    
    required_columns = ['hostname', 'rule_id', 'timestamp', 'count', 'source_label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check for invalid data types
    if df['rule_id'].isna().any():
        logger.error("Found NaN values in rule_id column")
        return False
    
    if df['count'].isna().any():
        logger.error("Found NaN values in count column")
        return False
    
    logger.info("Data validation passed")
    return True


if __name__ == "__main__":
    # Test the data loader
    test_config = {
        'training_data_path': './Training_data',
        'attack_data_path': './Training_data'
    }
    
    try:
        df = load_data('train', test_config)
        print(f"Loaded {len(df)} records")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Sample data:\n{df.head()}")
    except Exception as e:
        print(f"Error: {e}")