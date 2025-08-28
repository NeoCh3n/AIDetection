#!/usr/bin/env python3
"""
Test module for rule discovery and dynamic feature vector generation.
Validates rule detection from Qradar_rule folder and integration with Training_data.
"""

import os
import sys
import pandas as pd
import numpy as np
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_utils.rule_manager import RuleManager
from pipeline.feature_generator import FeatureGenerator

class TestRuleDiscovery(unittest.TestCase):
    """Test suite for dynamic rule discovery and feature generation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        import glob
        
        # Find actual rules directory using glob
        rule_dirs = glob.glob("*Qradar_rule")
        if rule_dirs:
            cls.rules_folder = rule_dirs[0]
        else:
            cls.rules_folder = "Qradar_rule"
            
        training_dirs = glob.glob("*Training_data")
        if training_dirs:
            cls.training_data_folder = training_dirs[0]
        else:
            cls.training_data_folder = "Training_data"
            
        cls.rule_manager = RuleManager(cls.rules_folder)
        cls.feature_generator = FeatureGenerator(cls.rules_folder)
    
    def test_rule_files_exist(self):
        """Test that rule files exist and are accessible."""
        base_path = self.rules_folder.replace('*', '')
        self.assertTrue(os.path.exists(base_path), f"Rules folder should exist: {base_path}")
        
        csv_files = [f for f in os.listdir(base_path) if f.endswith('.csv')]
        self.assertGreater(len(csv_files), 0, "Should find at least one CSV rule file")
    
    def test_rule_discovery(self):
        """Test rule discovery from CSV files."""
        rule_list = self.rule_manager.discover_rules(use_cache=False)
        
        self.assertIsInstance(rule_list, list, "Should return a list of rule IDs")
        self.assertGreater(len(rule_list), 0, "Should discover at least one rule")
        
        # Check that all elements are integers
        for rule_id in rule_list:
            self.assertIsInstance(rule_id, int, f"Rule ID {rule_id} should be an integer")
        
        # Check that list is sorted
        self.assertEqual(rule_list, sorted(rule_list), "Rule list should be sorted")
    
    def test_rule_mapping(self):
        """Test rule-to-index mapping."""
        rule_list = self.rule_manager.get_rule_list()
        rule_to_index = self.rule_manager.get_rule_to_index_map()
        
        self.assertEqual(len(rule_list), len(rule_to_index), 
                        "Rule list and mapping should have same length")
        
        # Test that all rules are in mapping
        for rule_id in rule_list:
            self.assertIn(rule_id, rule_to_index, f"Rule {rule_id} should be in mapping")
            self.assertIsInstance(rule_to_index[rule_id], int, 
                                f"Index for rule {rule_id} should be an integer")
    
    def test_vector_dimension(self):
        """Test that vector dimension matches actual rule count."""
        rule_list = self.rule_manager.get_rule_list()
        vector_dimension = self.rule_manager.get_vector_dimension()
        
        self.assertEqual(len(rule_list), vector_dimension,
                        "Vector dimension should match rule count")
        self.assertGreater(vector_dimension, 0, "Vector dimension should be positive")
    
    def test_feature_generator_initialization(self):
        """Test feature generator initialization."""
        self.feature_generator.initialize_rules()
        
        self.assertIsNotNone(self.feature_generator._rule_to_index,
                           "Rule mapping should be initialized")
        self.assertIsNotNone(self.feature_generator._vector_dimension,
                           "Vector dimension should be initialized")
    
    def test_feature_vector_creation(self):
        """Test feature vector creation with sample data."""
        self.feature_generator.initialize_rules()
        
        # Create sample aggregated data
        rule_list = self.rule_manager.get_rule_list()
        if len(rule_list) > 0:
            sample_rules = {rule_list[0]: 5, rule_list[1]: 3}
            
            df_sample = pd.DataFrame([{
                'window_id': '2025-07-29T09:30:00',
                'hostname': 'test-host',
                'aggregated_rules_dict': sample_rules,
                'label': 0
            }])
            
            X, y = self.feature_generator.generate_feature_vectors(df_sample, mode='train')
            
            self.assertEqual(X.shape[0], 1, "Should create 1 sample")
            self.assertEqual(X.shape[1], len(rule_list), 
                           f"Feature dimension should be {len(rule_list)}")
            self.assertEqual(y[0], 0, "Label should be 0")
    
    def test_rule_statistics(self):
        """Test rule statistics reporting."""
        self.feature_generator.initialize_rules()
        
        stats = self.feature_generator.get_rule_statistics()
        
        self.assertIn('total_rules', stats)
        self.assertIn('rule_min', stats)
        self.assertIn('rule_max', stats)
        
        self.assertGreater(stats['total_rules'], 0)
        self.assertLessEqual(stats['rule_min'], stats['rule_max'])
    
    def test_training_data_integration(self):
        """Test integration with actual training data."""
        base_path = self.training_data_folder.replace('*', '')
        
        if os.path.exists(base_path):
            # Test with various training data files
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if file.endswith('.csv'):
                        file_path = os.path.join(root, file)
                        try:
                            df = pd.read_csv(file_path)
                            
                            # Check if file has expected columns
                            expected_columns = ['sysmon_hostname (custom)', 'Custom Rule', 
                                              'Log Source Time (Minimum)', 'Count']
                            
                            if all(col in df.columns for col in expected_columns):
                                # Test rule ID extraction
                                rule_ids = df['Custom Rule'].dropna().astype(int).unique()
                                
                                # Verify rules are in discovered set
                                discovered_rules = set(self.rule_manager.get_rule_list())
                                present_rules = set(rule_ids).intersection(discovered_rules)
                                
                                self.assertGreater(len(present_rules), 0,
                                                 f"Should find matching rules in {file}")
                                
                        except Exception as e:
                            print(f"Warning: Could not process {file}: {e}")
    
    def test_rule_coverage_validation(self):
        """Test rule coverage validation."""
        self.feature_generator.initialize_rules()
        
        # Create test data with known rules
        rule_list = self.rule_manager.get_rule_list()
        if len(rule_list) > 0:
            test_df = pd.DataFrame([
                {
                    'window_id': 'test1',
                    'hostname': 'host1',
                    'aggregated_rules_dict': {rule_list[0]: 1, 999999: 2},  # 999999 is unknown
                    'label': 0
                }
            ])
            
            coverage = self.feature_generator.validate_rule_coverage(test_df)
            
            self.assertIn('present_rules', coverage)
            self.assertIn('missing_rules', coverage)
            self.assertIn('unknown_rules', coverage)
            
            self.assertEqual(coverage['present_rules'], 1)
            self.assertEqual(coverage['unknown_rules'], 1)
    
    def test_cache_functionality(self):
        """Test rule caching functionality."""
        # Clear cache
        self.rule_manager.clear_cache()
        
        # Discover rules without cache
        rules1 = self.rule_manager.discover_rules(use_cache=False)
        
        # Discover with cache (should use cached version)
        rules2 = self.rule_manager.discover_rules(use_cache=True)
        
        self.assertEqual(rules1, rules2, "Cached rules should match original discovery")
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        self.feature_generator.initialize_rules()
        
        empty_df = pd.DataFrame()
        X, y = self.feature_generator.generate_feature_vectors(empty_df, mode='train')
        
        self.assertEqual(X.shape[0], 0, "Empty DataFrame should produce empty matrix")
        self.assertIsNone(y, "Empty DataFrame should produce None labels")

if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)