#!/usr/bin/env python3
"""
Simple test script to verify pipeline integration
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add necessary paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared_utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Training_data'))

def test_training_mode():
    """Test training mode with existing CSV data"""
    print("Testing training mode...")
    
    try:
        # Check if training data exists
        training_path = "../Training_data"
        if os.path.exists(training_path):
            print(f"✓ Training data directory exists: {training_path}")
            
            # Check for CSV files
            csv_files = []
            for root, dirs, files in os.walk(training_path):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
            
            print(f"✓ Found {len(csv_files)} CSV files:")
            for csv_file in csv_files:
                print(f"  - {csv_file}")
                
                # Test reading a sample
                try:
                    df = pd.read_csv(csv_file, nrows=5)
                    print(f"    Sample columns: {list(df.columns)}")
                    print(f"    Sample shape: {df.shape}")
                except Exception as e:
                    print(f"    Error reading: {e}")
        else:
            print("✗ Training data directory not found")
            return False
            
        print("✓ Training mode test completed")
        return True
        
    except Exception as e:
        print(f"✗ Training mode test failed: {e}")
        return False

def test_detection_mode():
    """Test detection mode setup"""
    print("\nTesting detection mode...")
    
    try:
        # Check API integration files
        api_path = "../api_integration"
        if os.path.exists(api_path):
            print(f"✓ API integration directory exists: {api_path}")
            
            api_files = [
                "create_searches_Qradar.py",
                "status_searches_Qradar.py",
                "result_searches_Qradar.py",
                "delete_searches_Qradar.py"
            ]
            
            for file in api_files:
                file_path = os.path.join(api_path, file)
                if os.path.exists(file_path):
                    print(f"  ✓ {file}")
                else:
                    print(f"  ✗ {file} not found")
        
        # Check MongoDB files
        mongodb_path = "../mongodb"
        if os.path.exists(mongodb_path):
            print(f"✓ MongoDB directory exists: {mongodb_path}")
            
            mongodb_files = [
                "insert_DB.py",
                "query_DB.py",
                "delete_DB.py",
                "mongodb_config.json"
            ]
            
            for file in mongodb_files:
                file_path = os.path.join(mongodb_path, file)
                if os.path.exists(file_path):
                    print(f"  ✓ {file}")
                else:
                    print(f"  ✗ {file} not found")
        
        print("✓ Detection mode test completed")
        return True
        
    except Exception as e:
        print(f"✗ Detection mode test failed: {e}")
        return False

def test_shared_utils():
    """Test shared utilities"""
    print("\nTesting shared utilities...")
    
    try:
        # Test time_utils import
        from shared_utils.time_utils import parse_qradar_timestamp
        print("✓ time_utils import successful")
        
        # Test rule_manager import
        from shared_utils.rule_manager import get_rule_list
        print("✓ rule_manager import successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Shared utilities test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Pipeline Integration Test ===")
    print(f"Working directory: {os.getcwd()}")
    
    # Test training mode
    training_ok = test_training_mode()
    
    # Test detection mode
    detection_ok = test_detection_mode()
    
    # Test shared utilities
    utils_ok = test_shared_utils()
    
    print(f"\n=== Test Summary ===")
    print(f"Training mode: {'✓' if training_ok else '✗'}")
    print(f"Detection mode: {'✓' if detection_ok else '✗'}")
    print(f"Shared utils: {'✓' if utils_ok else '✗'}")
    
    all_passed = training_ok and detection_ok and utils_ok
    print(f"Overall: {'✓ All tests passed' if all_passed else '✗ Some tests failed'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)