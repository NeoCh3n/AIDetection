"""
Feature Aggregator Module - 30-minute window aggregation

This module aggregates individual rule triggers into 30-minute time windows,
creating the feature vectors needed for the Random Forest classifier.
It handles both training and detection modes with consistent aggregation logic.
"""

import pandas as pd
import numpy as np
import math
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, cast
import logging

# Ensure project root is on sys.path so "shared_utils" imports resolve when tests
# execute from nested directories (e.g., tests/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared_utils.time_utils import get_window_id, get_window_start_end

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def aggregate_to_windows(df: pd.DataFrame, window_size_minutes: int = 30, mode: str = 'train') -> pd.DataFrame:
    """
    Aggregate individual rule triggers into 30-minute time windows.
    
    This function takes raw event data and aggregates it into time windows
    suitable for machine learning features. Each row represents a 30-minute
    window for a specific hostname.
    
    Args:
        df: DataFrame with columns: hostname, rule_id, timestamp, count, source_label
        window_size_minutes: Size of time windows in minutes (default: 30)
        
    Returns:
        DataFrame with aggregated features per window:
        - window_id: Unique identifier for the 30-minute window
        - hostname: Host identifier
        - aggregated_rules: Dict[str, int] mapping rule_id -> raw integer total_count
        - total_events: Total number of events in this window
        - unique_rules: Number of unique rules triggered
        - source_label: Original data source ('normal' or 'attack')
        - window_start: Start time of the window
        - window_end: End time of the window
        - aggregated_rules_log1p_sum: Optional Dict[str, float] of log1p-summed counts
        - total_events_log1p: Optional float, sum of log1p(count) over events
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to aggregator")
        return pd.DataFrame()
    
    # Validate required columns
    required_columns = ['hostname', 'rule_id', 'timestamp', 'count', 'source_label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    logger.info(f"Aggregating {len(df)} events into {window_size_minutes}-minute windows...")
    
    # Normalize Source IP column if present (support both 'Source IP' and 'source_ip')
    if 'Source IP' in df.columns:
        df['source_ip'] = df['Source IP'].astype(str)
    elif 'source_ip' in df.columns:
        df['source_ip'] = df['source_ip'].astype(str)
    else:
        df['source_ip'] = '0.0.0.0'
    
    # Ensure timestamp column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Ensure numeric type for counts and keep raw integer counts for aggregation
    # Robust to objects/strings, NaNs, and negative values.
    df['count'] = pd.to_numeric(df['count'], errors='coerce')
    df['count'] = df['count'].where(df['count'] >= 0, 0)  # handle negatives
    df['count'] = df['count'].fillna(0)
    # Use integer counts for aggregation output to maintain interpretability and
    # satisfy tests; also compute log1p counts for optional downstream use.
    df['count'] = df['count'].astype(int)
    # Use math.log1p via Series.apply to satisfy static type checkers
    df['count_log1p'] = df['count'].astype(float).apply(math.log1p)
    
    # Generate window IDs for each event
    df['window_id'] = df['timestamp'].apply(
        lambda ts: get_window_id(ts, window_size_minutes)
    )
    
    # Group by window_id, hostname, source_label and source_ip for aggregation
    grouped = df.groupby(['window_id', 'hostname', 'source_label', 'source_ip'])
    
    # Aggregate rule counts into dictionaries
    aggregated_data = []
    
    for group_key, group_df in grouped:
        # Use typing.cast to explicitly type the tuple unpacking
        window_id, hostname, source_label, source_ip = cast(Tuple[str, str, str, str], group_key)
        group = group_df
        # Build rule count dictionary (RAW integer sums for interpretability/tests)
        rule_counts_raw = group.groupby('rule_id')['count'].sum().astype(int).to_dict()
        # Optional: also compute log1p-summed counts for modeling insights
        rule_counts_log = group.groupby('rule_id')['count_log1p'].sum().to_dict()
        
        # Convert rule_id keys to strings for JSON compatibility
        rule_counts_str: Dict[str, int] = {str(k): int(v) for k, v in rule_counts_raw.items()}
        rule_counts_log_str = {str(k): float(v) for k, v in rule_counts_log.items()}
        
        # Calculate additional metrics
        total_events = int(group['count'].sum())
        total_events_log1p = float(group['count_log1p'].sum())
        unique_rules = len(rule_counts_raw)
        
        # Convert source_label to binary is_attack
        is_attack = 1 if source_label == 'attack' else 0
        
        # Get window boundaries
        first_timestamp = group['timestamp'].min()
        window_start, window_end = get_window_start_end(first_timestamp, window_size_minutes)
        
        # Build result with mode-specific columns
        result_row = {
            'window_id': window_id,
            'hostname': hostname,
            'aggregated_rules': rule_counts_str,
            'total_events': int(total_events),
            'unique_rules': int(unique_rules),
            'window_start': window_start,
            'window_end': window_end,
            'source_label': source_label,
            'source_ip': source_ip
        }
        # Attach optional log1p-sum metrics for downstream use (not used by tests)
        result_row['aggregated_rules_log1p_sum'] = rule_counts_log_str
        result_row['total_events_log1p'] = total_events_log1p
        
        # Add label column for training mode
        if mode == 'train':
            result_row['is_attack'] = is_attack
        
        aggregated_data.append(result_row)
    
    # Create DataFrame from aggregated data
    result_df = pd.DataFrame(aggregated_data)

    # Backward/forward compatibility: also expose 'aggregated_rules_dict'
    if not result_df.empty and 'aggregated_rules' in result_df.columns:
        result_df['aggregated_rules_dict'] = result_df['aggregated_rules']
    
    if not result_df.empty:
        logger.info(f"Created {len(result_df)} aggregated windows")
        logger.info(f"Average events per window: {result_df['total_events'].mean():.2f}")
        logger.info(f"Average unique rules per window: {result_df['unique_rules'].mean():.2f}")
    
    return result_df


def validate_aggregated_data(df: pd.DataFrame) -> bool:
    """
    Validate the aggregated data meets requirements.
    
    Args:
        df: Aggregated DataFrame to validate
        
    Returns:
        True if data is valid, False otherwise
    """
    if df.empty:
        logger.error("Aggregated data is empty")
        return False
    
    required_columns = [
        'window_id', 'hostname', 'aggregated_rules', 
        'total_events', 'unique_rules'
    ]
    
    # Add mode-specific required columns
    if 'is_attack' in df.columns:
        required_columns.append('is_attack')
    else:
        required_columns.append('source_label')
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Validate data types and values
    if not all(isinstance(rules, dict) for rules in df['aggregated_rules']):
        logger.error("aggregated_rules column contains non-dictionary values")
        return False
    
    if not all(isinstance(total, (int, float)) for total in df['total_events']):
        logger.error("total_events column contains non-integer values")
        return False
    
    if not all(isinstance(unique, int) for unique in df['unique_rules']):
        logger.error("unique_rules column contains non-integer values")
        return False
    
    # Validate label columns
    if 'is_attack' in df.columns:
        valid_labels = df['is_attack'].isin([0, 1])
        if not valid_labels.all():
            logger.error("is_attack column contains invalid values (must be 0 or 1)")
            return False
    elif 'source_label' in df.columns:
        valid_labels = df['source_label'].isin(['normal', 'attack'])
        if not valid_labels.all():
            logger.error("source_label column contains invalid values")
            return False
    
    logger.info("Aggregated data validation passed")
    return True


def get_window_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate statistics about the aggregated windows.
    
    Args:
        df: Aggregated DataFrame
        
    Returns:
        Dictionary with statistics about the aggregation
    """
    if df.empty:
        return {}
    
    stats = {
        'total_windows': len(df),
        'unique_hosts': df['hostname'].nunique(),
        'avg_events_per_window': float(df['total_events'].mean()),
        'avg_unique_rules_per_window': float(df['unique_rules'].mean()),
        'avg_rules_per_window': float(df['aggregated_rules'].apply(len).mean()),
        'min_events': int(df['total_events'].min()),
        'max_events': int(df['total_events'].max()),
        'min_unique_rules': int(df['unique_rules'].min()),
        'max_unique_rules': int(df['unique_rules'].max())
    }
    
    # Add label statistics based on available columns
    if 'is_attack' in df.columns:
        stats['normal_windows'] = int(len(df[df['is_attack'] == 0]))
        stats['attack_windows'] = int(len(df[df['is_attack'] == 1]))
        stats['attack_ratio'] = float(df['is_attack'].mean())
    elif 'source_label' in df.columns:
        stats['normal_windows'] = int(len(df[df['source_label'] == 'normal']))
        stats['attack_windows'] = int(len(df[df['source_label'] == 'attack']))
    
    # Calculate rule frequency distribution
    all_rules = []
    for rules_dict in df['aggregated_rules']:
        all_rules.extend([int(rule_id) for rule_id in rules_dict.keys()])
    
    if all_rules:
        rule_counts = pd.Series(all_rules).value_counts()
        stats['most_common_rules'] = rule_counts.head(10).to_dict()
        stats['total_unique_rules'] = len(rule_counts)
    
    return stats


def save_aggregated_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save aggregated data to CSV file.
    
    Args:
        df: Aggregated DataFrame to save
        output_path: Path to save the file
    """
    if df.empty:
        logger.warning("No data to save")
        return
    
    # Convert aggregated_rules to JSON string for CSV storage
    df_to_save = df.copy()
    df_to_save['aggregated_rules'] = df_to_save['aggregated_rules'].apply(
        lambda x: str(x) if isinstance(x, dict) else str(x)
    )
    
    # Ensure datetime columns are properly formatted
    for col in ['window_start', 'window_end']:
        if col in df_to_save.columns:
            df_to_save[col] = df_to_save[col].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    df_to_save.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} aggregated windows to {output_path}")


def load_aggregated_data(input_path: str) -> pd.DataFrame:
    """
    Load aggregated data from CSV file.
    
    Args:
        input_path: Path to the CSV file
        
    Returns:
        DataFrame with aggregated data
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    
    df = pd.read_csv(input_path)
    
    # Convert aggregated_rules back to dictionary
    import ast
    df['aggregated_rules'] = df['aggregated_rules'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    # Convert datetime columns
    for col in ['window_start', 'window_end']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    logger.info(f"Loaded {len(df)} aggregated windows from {input_path}")
    return df


if __name__ == "__main__":
    # Test the aggregator with sample data
    import os
    
    # Create sample test data
    test_data = [
        {'hostname': 'DESKTOP-01', 'rule_id': 100001, 'timestamp': '2025-07-29 09:15:00', 'count': 5, 'source_label': 'normal'},
        {'hostname': 'DESKTOP-01', 'rule_id': 100002, 'timestamp': '2025-07-29 09:20:00', 'count': 3, 'source_label': 'normal'},
        {'hostname': 'DESKTOP-01', 'rule_id': 100001, 'timestamp': '2025-07-29 09:25:00', 'count': 2, 'source_label': 'normal'},
        {'hostname': 'DESKTOP-01', 'rule_id': 100003, 'timestamp': '2025-07-29 09:35:00', 'count': 8, 'source_label': 'attack'},
        {'hostname': 'DESKTOP-02', 'rule_id': 100001, 'timestamp': '2025-07-29 09:15:00', 'count': 1, 'source_label': 'normal'},
    ]
    
    test_df = pd.DataFrame(test_data)
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    
    # Run aggregation with training mode
    aggregated = aggregate_to_windows(test_df, mode='train')
    
    if not aggregated.empty:
        print("\n" + "="*60)
        print("FEATURE AGGREGATOR TEST RESULTS")
        print("="*60)
        print(f"Input events: {len(test_df)}")
        print(f"Output windows: {len(aggregated)}")
        print(f"Unique hosts: {aggregated['hostname'].nunique()}")
        if 'is_attack' in aggregated.columns:
            print(f"Label distribution: {dict(aggregated['is_attack'].value_counts())}")
        
        print("\nAggregated windows:")
        cols_to_show = ['window_id', 'hostname', 'total_events', 'unique_rules']
        if 'is_attack' in aggregated.columns:
            cols_to_show.append('is_attack')
        print(aggregated[cols_to_show])
        
        # Show detailed rule counts for first window
        if len(aggregated) > 0:
            first_window = aggregated.iloc[0]
            print(f"\nRule counts for window {first_window['window_id']}:")
            print(first_window['aggregated_rules'])
        
        # Calculate and display statistics
        stats = get_window_statistics(aggregated)
        print(f"\nWindow statistics:")
        for key, value in stats.items():
            if key != 'most_common_rules':
                print(f"  {key}: {value}")
        
        # Test detection mode
        print("\n" + "-"*30)
        print("TESTING DETECTION MODE")
        print("-"*30)
        aggregated_detect = aggregate_to_windows(test_df, mode='detect')
        print(f"Detection mode windows: {len(aggregated_detect)}")
        print(f"Columns in detection mode: {list(aggregated_detect.columns)}")
        
        # Save test data
        test_output = os.path.join(os.path.dirname(__file__), '../tests/test_aggregated_data.csv')
        os.makedirs(os.path.dirname(test_output), exist_ok=True)
        save_aggregated_data(aggregated, test_output)
        print(f"\nTest data saved to: {test_output}")
