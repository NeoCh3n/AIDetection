#!/usr/bin/env python3
"""
Detailed test script for data_loader.py - Pandas DataFrame validation
"""

import sys
import os
sys.path.append('./pipeline')

from data_loader import load_data
import pandas as pd

def test_data_loader_detailed():
    """Test data loader with detailed Pandas DataFrame output"""
    
    import glob
    training_dirs = glob.glob('./*Training_data*/normal')
    attack_dirs = glob.glob('./*Training_data*/attack')
    
    training_data_path = training_dirs[0] if training_dirs else './Training_data/normal'
    attack_data_path = attack_dirs[0] if attack_dirs else './Training_data/attack'
    
    config = {
        'training_data_path': training_data_path,
        'attack_data_path': attack_data_path
    }
    
    try:
        df = load_data('train', config)
        
        print("TESTING PANDAS DATAFRAME GENERATION")
        print("=" * 60)
        
        # Test 1: Verify DataFrame creation
        print(f"DataFrame Type: {type(df)}")
        print(f"Is Pandas DataFrame: {isinstance(df, pd.DataFrame)}")
        print(f"DataFrame Shape: {df.shape}")
        print()
        
        # Test 2: Detailed DataFrame structure
        print("DATAFRAME STRUCTURE:")
        print("-" * 30)
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print(f"Column names: {list(df.columns)}")
        print()
        
        # Test 3: Data types
        print("COLUMN DATA TYPES:")
        print("-" * 30)
        for col, dtype in df.dtypes.items():
            print(f"{col}: {dtype}")
        print()
        
        # Test 4: First row as example
        if not df.empty:
            print("EXAMPLE DATAFRAME ENTRY:")
            print("-" * 30)
            example = df.iloc[0]
            print(f"Index: {example.name}")
            print(f"hostname: '{example['hostname']}'")
            print(f"rule_id: {example['rule_id']} (type: {type(example['rule_id'])})")
            print(f"timestamp: {example['timestamp']} (type: {type(example['timestamp'])})")
            print(f"count: {example['count']} (type: {type(example['count'])})")
            print(f"source_label: '{example['source_label']}'")
            print()
            
            # Test 5: DataFrame preview
            print("FIRST 3 ROWS:")
            print("-" * 30)
            print(df.head(3))
            print()
            
            # Test 6: Data summary
            print("DATA SUMMARY:")
            print("-" * 30)
            print(f"Unique hosts: {df['hostname'].nunique()}")
            print(f"Unique rule IDs: {df['rule_id'].nunique()}")
            print(f"Total count sum: {df['count'].sum()}")
            print(f"Data sources: {dict(df['source_label'].value_counts())}")
            
            # Test 7: Timestamp processing verification
            print("TIMESTAMP PROCESSING VERIFICATION:")
            print("-" * 40)
            if len(df) > 0:
                print(f"Timestamp type: {type(df['timestamp'].iloc[0])}")
                print(f"First timestamp: {df['timestamp'].iloc[0]}")
                print(f"Min timestamp: {df['timestamp'].min()}")
                print(f"Max timestamp: {df['timestamp'].max()}")
                
                # Test boundary adjustment with specific examples
                sample_times = df['timestamp'].head(5)
                for i, ts in enumerate(sample_times):
                    print(f"Sample {i+1}: {ts.strftime('%b %d, %Y, %I:%M:%S %p')}")
        else:
            print("DataFrame is empty!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_loader_detailed()