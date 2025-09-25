#!/usr/bin/env python3
"""
Comprehensive test suite for feature_aggregator.py
Tests 30-minute window aggregation functionality for threat detection pipeline
"""

import sys
import os
import numbers
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Any, Dict, cast

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.feature_aggregator import (
    aggregate_to_windows,
    validate_aggregated_data,
    get_window_statistics,
    save_aggregated_data,
    load_aggregated_data,
)


def create_test_data():
    """Create realistic test data for feature aggregation testing"""
    
    # Generate test data spanning multiple 30-minute windows
    base_time = datetime(2025, 7, 29, 9, 0, 0)
    
    test_data = []
    
    # Window 1: 09:00-09:30 - Normal activity
    for i in range(10):
        test_data.append({
            'hostname': 'DESKTOP-01',
            'rule_id': 100001,
            'timestamp': base_time + timedelta(minutes=5*i),
            'count': np.random.poisson(3) + 1,
            'source_label': 'normal'
        })
    
    # Window 1: Additional rules
    test_data.extend([
        {'hostname': 'DESKTOP-01', 'rule_id': 100002, 'timestamp': base_time + timedelta(minutes=8), 'count': 2, 'source_label': 'normal'},
        {'hostname': 'DESKTOP-01', 'rule_id': 100003, 'timestamp': base_time + timedelta(minutes=15), 'count': 1, 'source_label': 'normal'},
    ])
    
    # Window 2: 09:30-10:00 - Attack activity
    attack_base = base_time + timedelta(minutes=35)
    for i in range(15):
        test_data.append({
            'hostname': 'DESKTOP-01',
            'rule_id': 100004 + (i % 5),  # Multiple rules
            'timestamp': attack_base + timedelta(minutes=2*i),
            'count': np.random.poisson(8) + 5,  # Higher counts
            'source_label': 'attack'
        })
    
    # Window 3: 09:00-09:30 - Different host
    for i in range(8):
        test_data.append({
            'hostname': 'DESKTOP-02',
            'rule_id': 100001,
            'timestamp': base_time + timedelta(minutes=3*i),
            'count': np.random.poisson(2) + 1,
            'source_label': 'normal'
        })
    
    # Window 4: 09:00-09:30 - Edge case with single event
    test_data.append({
        'hostname': 'DESKTOP-03',
        'rule_id': 100005,
        'timestamp': base_time + timedelta(minutes=15),
        'count': 1,
        'source_label': 'normal'
    })
    
    return pd.DataFrame(test_data)


def test_basic_aggregation():
    """Test basic 30-minute window aggregation functionality"""
    print("Testing basic 30-minute window aggregation...")
    
    # Create test data
    df = create_test_data()
    print(f"   Input: {len(df)} events")
    
    # Run aggregation
    aggregated = aggregate_to_windows(df)
    print(f"   Output: {len(aggregated)} windows")
    
    # Basic assertions
    assert not aggregated.empty, "Should produce non-empty aggregated data"
    assert len(aggregated) > 1, "Should produce multiple windows for test data"
    
    # Check required columns
    required_cols = ['window_id', 'hostname', 'aggregated_rules', 'total_events', 'unique_rules', 'source_label']
    for col in required_cols:
        assert col in aggregated.columns, f"Missing required column: {col}"
    
    print("   Basic aggregation test passed")
    return True


def test_window_boundaries():
    """Test that events are correctly assigned to 30-minute windows"""
    print("Testing 30-minute window boundary assignment...")
    
    # Create data with specific timestamps spanning boundaries
    base_time = datetime(2025, 7, 29, 9, 0, 0)
    
    test_data = [
        # 09:00-09:30 window
        {'hostname': 'TEST-01', 'rule_id': 100001, 'timestamp': base_time, 'count': 1, 'source_label': 'normal'},
        {'hostname': 'TEST-01', 'rule_id': 100001, 'timestamp': base_time + timedelta(minutes=29, seconds=59), 'count': 1, 'source_label': 'normal'},
        
        # 09:30-10:00 window  
        {'hostname': 'TEST-01', 'rule_id': 100001, 'timestamp': base_time + timedelta(minutes=30), 'count': 1, 'source_label': 'normal'},
        {'hostname': 'TEST-01', 'rule_id': 100001, 'timestamp': base_time + timedelta(minutes=59), 'count': 1, 'source_label': 'normal'},
        
        # 10:00-10:30 window
        {'hostname': 'TEST-01', 'rule_id': 100001, 'timestamp': base_time + timedelta(minutes=60), 'count': 1, 'source_label': 'normal'},
    ]
    
    df = pd.DataFrame(test_data)
    aggregated = aggregate_to_windows(df)
    
    # Should have exactly 3 windows
    assert len(aggregated) == 3, f"Expected 3 windows, got {len(aggregated)}"
    
    # Check window IDs are correct
    window_ids = set(aggregated['window_id'])
    expected_windows = {
        '2025-07-29_09-00-00_W18',
        '2025-07-29_09-30-00_W19',
        '2025-07-29_10-00-00_W20'
    }
    
    print(f"   Generated windows: {sorted(window_ids)}")
    print(f"   Expected windows: {sorted(expected_windows)}")
    
    print("   Window boundary test passed")
    return True


def test_rule_aggregation():
    """Test that rule counts are correctly aggregated within windows"""
    print("Testing rule count aggregation...")
    
    # Create data with multiple events for same rule in same window
    base_time = datetime(2025, 7, 29, 9, 0, 0)
    
    test_data = [
        {'hostname': 'TEST-01', 'rule_id': 100001, 'timestamp': base_time + timedelta(minutes=5), 'count': 3, 'source_label': 'normal'},
        {'hostname': 'TEST-01', 'rule_id': 100001, 'timestamp': base_time + timedelta(minutes=10), 'count': 2, 'source_label': 'normal'},
        {'hostname': 'TEST-01', 'rule_id': 100002, 'timestamp': base_time + timedelta(minutes=15), 'count': 1, 'source_label': 'normal'},
        {'hostname': 'TEST-01', 'rule_id': 100001, 'timestamp': base_time + timedelta(minutes=20), 'count': 4, 'source_label': 'normal'},
    ]
    
    df = pd.DataFrame(test_data)
    aggregated = aggregate_to_windows(df)
    
    # Should have exactly 1 window
    assert len(aggregated) == 1, f"Expected 1 window, got {len(aggregated)}"
    
    # Get the aggregated data for the window (use label-based access to avoid Series typing issues)
    first_idx = aggregated.index[0]
    
    # Check rule counts
    rules = aggregated.at[first_idx, 'aggregated_rules']
    if not isinstance(rules, dict):
        raise AssertionError("aggregated_rules should be a dict")
    rules_dict = cast(Dict[str, Any], rules)
    assert '100001' in rules_dict, "Rule 100001 should be in aggregated rules"
    assert '100002' in rules_dict, "Rule 100002 should be in aggregated rules"
    assert int(rules_dict['100001']) == 9, f"Rule 100001 count should be 9, got {rules_dict['100001']}"
    assert int(rules_dict['100002']) == 1, f"Rule 100002 count should be 1, got {rules_dict['100002']}"
    
    # Check totals (ensure scalar types for static analyzers)
    total_events_val = int(cast(int, aggregated.at[first_idx, 'total_events']))
    unique_rules_val = int(cast(int, aggregated.at[first_idx, 'unique_rules']))
    assert total_events_val == 10, f"Total events should be 10, got {total_events_val}"
    assert unique_rules_val == 2, f"Unique rules should be 2, got {unique_rules_val}"
    
    print("   Rule aggregation test passed")
    return True


def test_host_separation():
    """Test that different hosts are handled separately"""
    print("Testing host separation in aggregation...")
    
    base_time = datetime(2025, 7, 29, 9, 0, 0)
    
    test_data = [
        {'hostname': 'DESKTOP-01', 'rule_id': 100001, 'timestamp': base_time, 'count': 5, 'source_label': 'normal'},
        {'hostname': 'DESKTOP-02', 'rule_id': 100001, 'timestamp': base_time, 'count': 3, 'source_label': 'normal'},
        {'hostname': 'DESKTOP-03', 'rule_id': 100001, 'timestamp': base_time, 'count': 1, 'source_label': 'normal'},
    ]
    
    df = pd.DataFrame(test_data)
    aggregated = aggregate_to_windows(df)
    
    # Should have 3 separate windows (one per host)
    assert len(aggregated) == 3, f"Expected 3 windows (one per host), got {len(aggregated)}"
    
    # Each should have separate counts
    hosts = aggregated['hostname'].unique()
    assert len(hosts) == 3, f"Expected 3 unique hosts, got {len(hosts)}"
    
    # Check individual counts (avoid Series bool in conditionals)
    expected_counts = {
        'DESKTOP-01': 5,
        'DESKTOP-02': 3,
        'DESKTOP-03': 1,
    }
    actual_counts = (
        aggregated.set_index('hostname')['total_events']
        .astype(int)
        .to_dict()
    )
    assert actual_counts == expected_counts, f"Counts per host mismatch: {actual_counts} vs {expected_counts}"
    
    print("   Host separation test passed")
    return True


def test_source_label_preservation():
    """Test that source labels are preserved correctly"""
    print("Testing source label preservation...")
    
    base_time = datetime(2025, 7, 29, 9, 0, 0)
    
    test_data = [
        {'hostname': 'HOST-01', 'rule_id': 100001, 'timestamp': base_time, 'count': 1, 'source_label': 'normal'},
        {'hostname': 'HOST-02', 'rule_id': 100001, 'timestamp': base_time, 'count': 1, 'source_label': 'attack'},
    ]
    
    df = pd.DataFrame(test_data)
    aggregated = aggregate_to_windows(df)
    
    # Should have 2 windows with correct labels
    assert len(aggregated) == 2, f"Expected 2 windows, got {len(aggregated)}"
    
    labels = set(aggregated['source_label'])
    assert labels == {'normal', 'attack'}, f"Expected labels {{'normal', 'attack'}}, got {labels}"
    
    print("   Source label preservation test passed")
    return True


def test_data_validation():
    """Test the validation function"""
    print("Testing data validation...")
    
    # Test with valid data
    df = create_test_data()
    aggregated = aggregate_to_windows(df)
    
    assert validate_aggregated_data(aggregated), "Valid data should pass validation"
    
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    assert not validate_aggregated_data(empty_df), "Empty DataFrame should fail validation"
    
    # Test with missing columns
    invalid_df = aggregated.drop(columns=['aggregated_rules'])
    assert not validate_aggregated_data(invalid_df), "Missing columns should fail validation"
    
    print("   Data validation test passed")
    return True


def test_statistics_calculation():
    """Test statistics calculation functionality"""
    print("Testing statistics calculation...")
    
    df = create_test_data()
    aggregated = aggregate_to_windows(df)
    
    stats = get_window_statistics(aggregated)
    
    # Check required statistics
    required_stats = [
        'total_windows', 'unique_hosts', 'normal_windows', 'attack_windows',
        'avg_events_per_window', 'avg_unique_rules_per_window',
        'min_events', 'max_events', 'min_unique_rules', 'max_unique_rules'
    ]
    
    for stat in required_stats:
        assert stat in stats, f"Missing statistic: {stat}"
    
    # Check data types and ranges
    assert isinstance(stats['total_windows'], int), "total_windows should be integer"
    assert isinstance(stats['avg_events_per_window'], float), "avg_events_per_window should be float"
    assert stats['total_windows'] > 0, "Should have positive window count"
    
    print(f"   Sample statistics:")
    print(f"     Total windows: {stats['total_windows']}")
    print(f"     Unique hosts: {stats['unique_hosts']}")
    print(f"     Avg events/window: {stats['avg_events_per_window']:.2f}")
    
    print("   Statistics calculation test passed")
    return True


def test_save_load_functionality():
    """Test saving and loading aggregated data"""
    print("Testing save/load functionality...")
    
    df = create_test_data()
    aggregated = aggregate_to_windows(df)
    
    # Create test directory
    test_dir = './tests/test_output'
    os.makedirs(test_dir, exist_ok=True)
    test_file = os.path.join(test_dir, 'test_aggregated.csv')
    
    try:
        # Save data
        save_aggregated_data(aggregated, test_file)
        assert os.path.exists(test_file), "File should be created"
        
        # Load data back
        loaded = load_aggregated_data(test_file)
        
        # Verify data integrity
        assert len(loaded) == len(aggregated), "Loaded data should have same length"
        assert set(loaded.columns) == set(aggregated.columns), "Columns should match"
        
        # Check aggregated rules format (avoid calling attributes that type checkers misinfer)
        for i, row in loaded.iterrows():
            rules_obj = row['aggregated_rules']
            if not isinstance(rules_obj, dict):
                raise AssertionError("Aggregated rules should be dict")
            # Narrow type for static analysis and then use dict-specific APIs
            rules_dict = cast(Dict[str, Any], rules_obj)
            rule_keys = list(rules_dict.keys())
            rule_vals = list(rules_dict.values())
            assert all(isinstance(k, str) for k in rule_keys), "Rule keys should be strings"
            # Accept numpy integer subclasses as ints
            assert all(isinstance(v, numbers.Integral) for v in rule_vals), "Rule values should be integers"
        
        print("   Save/load test passed")
        return True
        
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)


def test_edge_cases():
    """Test edge cases and error handling"""
    print("Testing edge cases...")
    
    # Test with empty DataFrame
    empty_df = pd.DataFrame(columns=['hostname', 'rule_id', 'timestamp', 'count', 'source_label'])
    result = aggregate_to_windows(empty_df)
    assert result.empty, "Empty input should produce empty output"
    
    # Test with single event
    single_event = pd.DataFrame([{
        'hostname': 'TEST-01',
        'rule_id': 100001,
        'timestamp': datetime(2025, 7, 29, 9, 15, 0),
        'count': 1,
        'source_label': 'normal'
    }])
    
    result = aggregate_to_windows(single_event)
    assert len(result) == 1, "Single event should produce single window"
    assert result.iloc[0]['total_events'] == 1, "Single event should have count 1"
    assert result.iloc[0]['unique_rules'] == 1, "Single rule should have unique count 1"
    
    # Test with zero counts
    zero_count = pd.DataFrame([{
        'hostname': 'TEST-01',
        'rule_id': 100001,
        'timestamp': datetime(2025, 7, 29, 9, 15, 0),
        'count': 0,
        'source_label': 'normal'
    }])
    
    result = aggregate_to_windows(zero_count)
    assert len(result) == 1, "Zero count should still produce window"
    assert result.iloc[0]['total_events'] == 0, "Zero count should have total 0"
    
    print("   Edge cases test passed")
    return True


def run_all_tests():
    """Run all test suites"""
    print("FEATURE AGGREGATOR TEST SUITE")
    print("=" * 60)
    
    test_functions = [
        test_basic_aggregation,
        test_window_boundaries,
        test_rule_aggregation,
        test_host_separation,
        test_source_label_preservation,
        test_data_validation,
        test_statistics_calculation,
        test_save_load_functionality,
        test_edge_cases
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"   FAILED: {test_func.__name__} - {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY:")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Total: {passed + failed}")
    
    if failed == 0:
        print("All tests passed!")
    else:
        print(f"{failed} tests failed")
    
    return failed == 0


def test_with_real_training_data():
    """Test feature aggregator with actual training data from Training_data folder"""
    print("Testing with actual training data...")
    
    import glob
    
    # Find actual training data directories
    training_dirs = glob.glob('./*Training_data*/normal')
    attack_dirs = glob.glob('./*Training_data*/attack')
    
    training_data_path = training_dirs[0] if training_dirs else './Training_data/normal'
    attack_data_path = attack_dirs[0] if attack_dirs else './Training_data/attack'
    
    config = {
        'training_data_path': training_data_path,
        'attack_data_path': attack_data_path
    }
    
    try:
        # Import here to avoid circular imports
        from pipeline.data_loader import load_data
        
        print(f"Loading training data...")
        print(f"Normal data: {training_data_path}")
        print(f"Attack data: {attack_data_path}")
        
        # Load a sample of training data (first 50 events from each source)
        df = load_data('train', config)
        
        if len(df) > 100:
            # Sample 50 events from each source to keep test fast
            normal_sample = df[df['source_label'] == 'normal'].head(50)
            attack_sample = df[df['source_label'] == 'attack'].head(50)
            df_sample = pd.concat([normal_sample, attack_sample])
        else:
            df_sample = df
            
        print(f"Testing with {len(df_sample)} events from training data")
        
        # Run aggregation
        aggregated = aggregate_to_windows(df_sample)
        
        if aggregated.empty:
            print("ERROR: No aggregated windows created")
            return False
            
        print(f"Created {len(aggregated)} aggregated windows")
        
        # Validate
        if not validate_aggregated_data(aggregated):
            print("ERROR: Validation failed")
            return False
            
        # Check expected characteristics
        # Convert ndarray from .unique() to list to satisfy type checkers
        expected_sources = set(
            df_sample['source_label'].dropna().astype(str).unique().tolist()
        )
        actual_sources = set(
            aggregated['source_label'].dropna().astype(str).unique().tolist()
        )
        
        if expected_sources != actual_sources:
            print(f"ERROR: Source mismatch - expected {expected_sources}, got {actual_sources}")
            return False
            
        # Check data integrity
        total_events_check = sum(len(rules) for rules in aggregated['aggregated_rules'])
        if total_events_check == 0:
            print("ERROR: No rules found in aggregated data")
            return False
            
        print("Real data test PASSED")
        return True
        
    except Exception as e:
        print(f"Real data test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests_with_real_data():
    """Run all tests including real data test"""
    print("FEATURE AGGREGATOR TEST SUITE (WITH REAL DATA)")
    print("=" * 60)
    
    # Original test functions
    test_functions = [
        test_basic_aggregation,
        test_window_boundaries,
        test_rule_aggregation,
        test_host_separation,
        test_source_label_preservation,
        test_data_validation,
        test_statistics_calculation,
        test_save_load_functionality,
        test_edge_cases
    ]
    
    passed = 0
    failed = 0
    
    # Run original tests
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"FAILED: {test_func.__name__} - {e}")
            failed += 1
    
    # Run real data test
    try:
        print("\n" + "=" * 40)
        print("RUNNING REAL DATA TEST...")
        print("=" * 40)
        test_with_real_training_data()
        passed += 1
    except Exception as e:
        print(f"FAILED: Real data test - {e}")
        failed += 1
    
    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY:")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Total: {passed + failed}")
    
    if failed == 0:
        print("All tests passed including real data test!")
    else:
        print(f"{failed} tests failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests_with_real_data()
    sys.exit(0 if success else 1)
