#!/usr/bin/env python3
"""
Test script for UAT-to-Production rule ID mapping functionality
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'shared_utils'))
sys.path.append(str(Path(__file__).parent.parent / 'pipeline'))

from shared_utils.qradar_rule_manager import QRadarRuleManager
from pipeline.feature_generator import FeatureGenerator

def test_uat_mapping():
    """Test UAT-to-Production rule mapping functionality."""
    print("=== Testing UAT-to-Production Rule Mapping ===\n")
    
    # Test 1: Load UAT mapping
    print("1. Testing UAT mapping file loading...")
    manager = QRadarRuleManager(environment='uat')
    uat_to_prod = manager.get_uat_to_prod_map()
    print(f"   Loaded {len(uat_to_prod)} UAT-to-Production mappings")
    
    # Show sample mappings
    sample_count = min(5, len(uat_to_prod))
    if sample_count > 0:
        print(f"   Sample mappings (showing {sample_count}):")
        for i, (uat, prod) in enumerate(list(uat_to_prod.items())[:sample_count]):
            print(f"   - UAT {uat} → Production {prod}")
    
    # Test 2: Production rule list
    print("\n2. Testing production rule list...")
    prod_rules = manager.get_production_rule_list()
    print(f"   Production baseline: {len(prod_rules)} rules")
    print(f"   Production rule range: {min(prod_rules)} to {max(prod_rules) if prod_rules else 0}")
    
    # Test 3: Production rule to index mapping
    print("\n3. Testing production rule to index mapping...")
    prod_to_idx = manager.get_production_rule_to_index_map()
    print(f"   Production rule indices: {len(prod_to_idx)} mappings")
    
    # Verify consistency
    if len(prod_rules) != len(prod_to_idx):
        print(f"   ERROR: Mismatch - {len(prod_rules)} rules vs {len(prod_to_idx)} indices")
    else:
        print("   ✓ Rule list and index mapping are consistent")
    
    # Test 4: Mapping validation
    print("\n4. Testing mapping validation...")
    validation_results = manager.validate_mapping_consistency()
    print(f"   Validation results:")
    print(f"   - Valid: {validation_results['valid']}")
    print(f"   - Warnings: {len(validation_results['warnings'])}")
    print(f"   - Errors: {len(validation_results['errors'])}")
    
    if validation_results['warnings']:
        print(f"   Warnings: {validation_results['warnings']}")
    if validation_results['errors']:
        print(f"   Errors: {validation_results['errors']}")
    
    if 'stats' in validation_results:
        stats = validation_results['stats']
        print(f"   - Total mappings: {stats.get('total_mappings', 0)}")
        print(f"   - Production rules: {stats.get('production_rules', 0)}")
        print(f"   - Coverage: {stats.get('coverage_percentage', 0):.1f}%")
    
    return validation_results['valid']

def test_feature_generator():
    """Test feature generator with UAT mapping."""
    print("\n=== Testing Feature Generator with UAT Mapping ===\n")
    
    # Test 1: Initialize feature generator for UAT
    print("1. Testing UAT feature generator...")
    uat_generator = FeatureGenerator(environment='uat')
    
    # Test 2: Initialize feature generator for Production
    print("2. Testing Production feature generator...")
    prod_generator = FeatureGenerator(environment='prod')
    
    # Test 3: Verify consistent vector dimensions
    print("\n3. Testing vector dimension consistency...")
    
    # Create sample aggregated data
    sample_data = [
        {
            'window_id': '2024-01-01T00:00:00',
            'hostname': 'test-host-1',
            'aggregated_rules_dict': {200033: 5, 200045: 3, 200099: 1},
            'label': 0
        },
        {
            'window_id': '2024-01-01T00:30:00',
            'hostname': 'test-host-2',
            'aggregated_rules_dict': {200033: 2, 200001: 8},
            'label': 1
        }
    ]
    
    df_sample = pd.DataFrame(sample_data)
    
    try:
        X_uat, y_uat = uat_generator.generate_feature_vectors(df_sample, mode='train')
        X_prod, y_prod = prod_generator.generate_feature_vectors(df_sample, mode='train')
        
        print(f"   UAT vector shape: {X_uat.shape}")
        print(f"   Production vector shape: {X_prod.shape}")
        
        if X_uat.shape[1] == X_prod.shape[1]:
            print("   ✓ Vector dimensions are consistent between environments")
        else:
            print(f"   ERROR: Dimension mismatch - UAT: {X_uat.shape[1]}, Prod: {X_prod.shape[1]}")
            return False
        
        # Test 4: Verify UAT rule mapping
        print("\n4. Testing UAT rule mapping in feature generation...")
        
        # Check if UAT rules were correctly mapped to production rules
        uat_manager = QRadarRuleManager(environment='uat')
        uat_to_prod = uat_manager.get_uat_to_prod_map()
        
        # Verify that mapped rules appear in the correct positions
        if 200033 in uat_to_prod:
            prod_200033 = uat_to_prod[200033]
            prod_idx = uat_generator._rule_to_index.get(prod_200033)
            if prod_idx is not None:
                print(f"   ✓ UAT rule 200033 correctly mapped to production rule {prod_200033} at index {prod_idx}")
            else:
                print(f"   WARNING: Production rule {prod_200033} not found in feature vector")
        
        return True
        
    except Exception as e:
        print(f"   ERROR in feature generation: {e}")
        return False

def test_cross_environment_consistency():
    """Test cross-environment consistency."""
    print("\n=== Testing Cross-Environment Consistency ===\n")
    
    # Test that UAT and Production environments produce same feature vectors
    # when given production data
    
    # Create production-style sample data
    prod_sample_data = [
        {
            'window_id': '2024-01-01T00:00:00',
            'hostname': 'prod-host-1',
            'aggregated_rules_dict': {100033: 5, 100045: 3, 100100: 1},
            'label': 0
        },
        {
            'window_id': '2024-01-01T00:30:00',
            'hostname': 'prod-host-2',
            'aggregated_rules_dict': {100033: 2, 100001: 8},
            'label': 1
        }
    ]
    
    df_prod = pd.DataFrame(prod_sample_data)
    
    # Test both environments with production data
    uat_gen = FeatureGenerator(environment='uat')
    prod_gen = FeatureGenerator(environment='prod')
    
    try:
        X_uat, y_uat = uat_gen.generate_feature_vectors(df_prod, mode='train')
        X_prod, y_prod = prod_gen.generate_feature_vectors(df_prod, mode='train')
        
        # Check if vectors are identical
        if np.array_equal(X_uat, X_prod):
            print("   ✓ Cross-environment consistency verified")
            return True
        else:
            print("   ERROR: Cross-environment inconsistency detected")
            print(f"   Max difference: {np.max(np.abs(X_uat - X_prod))}")
            return False
            
    except Exception as e:
        print(f"   ERROR in cross-environment testing: {e}")
        return False

def main():
    """Run all tests."""
    print("Starting UAT-to-Production Rule Mapping Tests...\n")
    
    # Change to project directory
    os.chdir(Path(__file__).parent.parent)
    
    all_passed = True
    
    try:
        # Test 1: UAT mapping
        test1_passed = test_uat_mapping()
        all_passed = all_passed and test1_passed
        
        # Test 2: Feature generator
        test2_passed = test_feature_generator()
        all_passed = all_passed and test2_passed
        
        # Test 3: Cross-environment consistency
        test3_passed = test_cross_environment_consistency()
        all_passed = all_passed and test3_passed
        
        print(f"\n{'='*50}")
        if all_passed:
            print("✅ All tests PASSED")
        else:
            print("❌ Some tests FAILED")
        print(f"{'='*50}")
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)