#!/usr/bin/env python3
"""
Unified Test Suite for API-to-MongoDB Data Flow

Comprehensive testing framework that combines standalone, simple, and comprehensive
approaches into a single test suite with conditional execution based on dependencies.

Testing Levels:
1. Standalone (Level 0): No dependencies, pure Python testing
2. Simple (Level 1): Mocked MongoDB, basic integration testing  
3. Comprehensive (Level 2): Full integration with real MongoDB (optional)

Python 3.6.8 Compatible
"""

import os
import sys
import json
import unittest
import tempfile
import shutil
import warnings
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add project paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mongodb'))

# Test level configuration
class TestConfig:
    """Configuration for test execution levels."""
    
    @classmethod
    def get_test_level(cls):
        """Determine test level based on available dependencies."""
        level = 0  # Default: standalone
        
        # Check for MongoDB dependencies
        try:
            import pymongo
            level = max(level, 1)
        except ImportError:
            print("Warning: pymongo not available, skipping MongoDB tests")
        
        # Check for project modules
        try:
            from mongodb.insert_DB import AQLDataInserter
            from mongodb.query_DB import AQLQueryManager
            level = max(level, 1)
        except ImportError as e:
            print(f"Warning: {e}, using mocked implementations")
        
        try:
            from mongodb.mongodb_connection import MongoDBConnectionManager
            level = max(level, 2)
        except ImportError as e:
            print(f"Warning: {e}, skipping comprehensive tests")
        
        return level

# Base standalone testing
class BaseStandaloneAPITest(unittest.TestCase):
    """Base standalone tests with zero dependencies."""
    
    def setUp(self):
        """Set up base test fixtures."""
        super().setUp()
        self.sample_aql_json = {
            "events": [
                {
                    "Custom Rule": "100227",
                    "Count": 50,
                    "sysmon_hostname (custom)": "test-host-1",
                    "Log Source Time (Minimum)": "Jul 29, 2025, 9:50:55 AM"
                },
                {
                    "Custom Rule": "100221",
                    "Count": 25,
                    "sysmon_hostname (custom)": "test-host-1",
                    "Log Source Time (Minimum)": "Jul 29, 2025, 9:55:55 AM"
                },
                {
                    "Custom Rule": "100227",
                    "Count": 75,
                    "sysmon_hostname (custom)": "test-host-2",
                    "Log Source Time (Minimum)": "Jul 29, 2025, 10:00:55 AM"
                }
            ]
        }
    
    class AQLDataInserterTest:
        """Standalone AQL processor for testing."""
        
        def __init__(self, config_path=None):
            self.config_path = config_path
            self.config = self._load_config()
        
        def _load_config(self):
            """Load test configuration."""
            return {
                "mongodb": {
                    "host": "localhost",
                    "port": 27017,
                    "db_name": "test_ransomware_detection"
                },
                "pipeline": {
                    "mode": "test",
                    "window_size_minutes": 30,
                    "retention_days": 1
                }
            }
        
        def parse_aql_timestamp(self, timestamp_str):
            """Parse QRadar AQL timestamp string to datetime."""
            try:
                return datetime.strptime(timestamp_str, "%b %d, %Y, %I:%M:%S %p")
            except ValueError:
                try:
                    return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    return None
        
        def get_window_boundaries(self, timestamp):
            """Calculate 30-minute window boundaries for AQL data."""
            minutes = timestamp.minute
            window_start = timestamp.replace(
                minute=(minutes // 30) * 30, 
                second=0, 
                microsecond=0
            )
            window_end = window_start + timedelta(minutes=30)
            window_id = window_start.strftime("%Y-%m-%d_%H-%M-%S")
            return window_id, window_start, window_end
        
        def parse_aql_json_result(self, result_data):
            """Parse QRadar AQL JSON results into detection windows."""
            if 'events' not in result_data:
                return []
            
            events = result_data['events']
            window_groups = {}
            
            for event in events:
                try:
                    rule_id = str(event.get('Custom Rule', '0'))
                    count = int(event.get('Count', 0))
                    hostname = str(event.get('sysmon_hostname (custom)', 'global'))
                    
                    event_time_str = event.get('Log Source Time (Minimum)', '')
                    if not event_time_str:
                        continue
                    
                    event_time = self.parse_aql_timestamp(event_time_str)
                    if not event_time:
                        continue
                    
                    window_id, window_start, window_end = self.get_window_boundaries(event_time)
                    
                    if window_id not in window_groups:
                        window_groups[window_id] = {
                            '_id': window_id,
                            'window_start': window_start,
                            'window_end': window_end,
                            'query_time': datetime.now(),
                            'feature_vector': {},
                            'host_triggers': {},
                            'total_triggers': 0,
                            'total_rules_triggered': 0,
                            'metadata': {
                                'source': 'qradar_aql',
                                'data_type': 'detection',
                                'window_size_minutes': 30
                            }
                        }
                    
                    window_groups[window_id]['feature_vector'][rule_id] = (
                        window_groups[window_id]['feature_vector'].get(rule_id, 0) + count
                    )
                    
                    if hostname not in window_groups[window_id]['host_triggers']:
                        window_groups[window_id]['host_triggers'][hostname] = {
                            'total_triggers': 0,
                            'rules': {}
                        }
                    
                    window_groups[window_id]['host_triggers'][hostname]['total_triggers'] += count
                    window_groups[window_id]['host_triggers'][hostname]['rules'][rule_id] = (
                        window_groups[window_id]['host_triggers'][hostname]['rules'].get(rule_id, 0) + count
                    )
                    
                    window_groups[window_id]['total_triggers'] += count
                    window_groups[window_id]['total_rules_triggered'] = len(
                        window_groups[window_id]['feature_vector']
                    )
                    
                except Exception as e:
                    continue
            
            return list(window_groups.values())

class Level0StandaloneTests(BaseStandaloneAPITest):
    """Level 0: Standalone tests with no dependencies."""
    
    def test_aql_json_parsing_standalone(self):
        """Test AQL JSON parsing functionality."""
        processor = self.AQLDataInserterTest()
        documents = processor.parse_aql_json_result(self.sample_aql_json)
        
        # Assertions
        self.assertEqual(len(documents), 1)
        self.assertIsNotNone(documents[0]['_id'])
        self.assertEqual(documents[0]['total_triggers'], 150)
        self.assertIn('test-host-1', documents[0]['host_triggers'])
        self.assertIn('test-host-2', documents[0]['host_triggers'])
        self.assertIn('100227', documents[0]['feature_vector'])
        self.assertIn('100221', documents[0]['feature_vector'])
        self.assertEqual(documents[0]['feature_vector']['100227'], 125)  # 50 + 75
        self.assertEqual(documents[0]['feature_vector']['100221'], 25)
    
    def test_timestamp_parsing_standalone(self):
        """Test AQL timestamp parsing."""
        processor = self.AQLDataInserterTest()
        
        # Test valid timestamp
        timestamp_str = "Jul 29, 2025, 9:50:55 AM"
        parsed = processor.parse_aql_timestamp(timestamp_str)
        self.assertIsNotNone(parsed)
        if parsed:  # Add null check to prevent None attribute access
            self.assertEqual(parsed.year, 2025)
            self.assertEqual(parsed.month, 7)
            self.assertEqual(parsed.day, 29)
            self.assertEqual(parsed.hour, 9)
            self.assertEqual(parsed.minute, 50)
        
        # Test invalid timestamp
        invalid_timestamp = "invalid-timestamp"
        parsed_invalid = processor.parse_aql_timestamp(invalid_timestamp)
        self.assertIsNone(parsed_invalid)
    
    def test_window_boundary_calculation_standalone(self):
        """Test 30-minute window boundary calculation."""
        processor = self.AQLDataInserterTest()
        
        test_time = datetime(2025, 7, 29, 9, 50, 55)
        window_id, window_start, window_end = processor.get_window_boundaries(test_time)
        
        # Should align to 9:30-10:00 window
        self.assertEqual(window_start, datetime(2025, 7, 29, 9, 30, 0))
        self.assertEqual(window_end, datetime(2025, 7, 29, 10, 0, 0))
        self.assertEqual(window_id, "2025-07-29_09-30-00")

class Level1MockTests(BaseStandaloneAPITest):
    """Level 1: Tests with mocked MongoDB dependencies."""
    
    @classmethod
    def setUpClass(cls):
        """Set up for mock tests."""
        super().setUpClass()
        cls.test_level = TestConfig.get_test_level()
        if cls.test_level < 1:
            raise unittest.SkipTest("Level 1 tests require pymongo")
    
    def setUp(self):
        """Set up mock test fixtures."""
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_path = os.path.join(self.temp_dir, 'test_config.json')
        
        # Create test configuration
        self.test_config = {
            "mongodb": {
                "host": "localhost",
                "port": 27017,
                "db_name": "test_ransomware_detection"
            },
            "pipeline": {
                "mode": "test",
                "window_size_minutes": 30,
                "retention_days": 1
            }
        }
        
        with open(self.test_config_path, 'w') as f:
            json.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up mock test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_mock_aql_processing_with_pymongo(self):
        """Test AQL processing with mocked MongoDB."""
        try:
            from mongodb.insert_DB import AQLDataInserter
        except ImportError:
            self.skipTest("AQLDataInserter not available")
        
        with patch('pymongo.MongoClient') as mock_mongo_client:
            # Setup mocks
            mock_client = Mock()
            mock_db = Mock()
            mock_collection = Mock()
            
            mock_mongo_client.return_value = mock_client
            mock_client.__getitem__.return_value = mock_db
            mock_db.__getitem__.return_value = mock_collection
            
            # Mock bulk write result
            mock_bulk_result = Mock()
            mock_bulk_result.upserted_count = 1
            mock_bulk_result.modified_count = 0
            mock_collection.bulk_write.return_value = mock_bulk_result
            
            # Test AQL processing
            inserter = AQLDataInserter(self.test_config_path)
            inserter.client = mock_client
            inserter.db = mock_db
            
            documents = inserter.parse_aql_json_result(self.sample_aql_json)
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0]['total_triggers'], 150)

class Level2IntegrationTests(BaseStandaloneAPITest):
    """Level 2: Full integration tests with real MongoDB."""
    
    @classmethod
    def setUpClass(cls):
        """Set up for integration tests."""
        super().setUpClass()
        cls.test_level = TestConfig.get_test_level()
        if cls.test_level < 2:
            raise unittest.SkipTest("Level 2 tests require full MongoDB setup")
    
    def setUp(self):
        """Set up integration test fixtures."""
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_path = os.path.join(self.temp_dir, 'test_integration_config.json')
        
        # Create comprehensive test configuration
        self.test_config = {
            "mongodb": {
                "host": "localhost",
                "port": 27017,
                "db_name": "test_ransomware_detection_integration",
                "connection_string": "mongodb://localhost:27017/"
            },
            "pipeline": {
                "mode": "test",
                "window_size_minutes": 30,
                "retention_days": 1
            },
            "collections": {
                "events": "test_events",
                "windows": "test_windows",
                "predictions": "test_predictions"
            }
        }
        
        with open(self.test_config_path, 'w') as f:
            json.dump(self.test_config, f, indent=2)
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @staticmethod
    def check_mongo_available():
        """Check if MongoDB is available for integration tests."""
        try:
            import pymongo
            client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            return True
        except Exception:
            return False
    
    def test_unified_pipeline_integration(self):
        """Test complete unified pipeline integration."""
        if not self.check_mongo_available():
            self.skipTest("MongoDB not available for integration tests")
        
        try:
            from mongodb.mongodb_connection import MongoDBConnectionManager
        except ImportError:
            self.skipTest("MongoDBConnectionManager not available")
        
        with MongoDBConnectionManager(self.test_config_path) as manager:
            # Test connection
            self.assertTrue(manager.connect())
            
            # Test data insertion and retrieval
            test_event = {
                "hostname": "test-integration-host",
                "rule_id": 999999,
                "timestamp": datetime.now(),
                "count": 100
            }
            
            success = manager.insert_event(test_event)
            self.assertTrue(success)
            
            # Test data summary
            summary = manager.get_data_summary()
            self.assertIsInstance(summary, dict)
            
            # Cleanup
            if manager.events_collection:
                manager.events_collection.delete_many({"hostname": "test-integration-host"})

class TestSuiteOrchestrator:
    """Orchestrates test suite execution based on available dependencies."""
    
    @classmethod
    def get_test_suite(cls):
        """Get appropriate test suite based on available dependencies."""
        test_level = TestConfig.get_test_level()
        
        # Base suite with standalone tests
        suite = unittest.TestSuite()
        
        # Always add Level 0 tests
        level0_tests = unittest.TestLoader().loadTestsFromTestCase(Level0StandaloneTests)
        suite.addTests(level0_tests)
        
        # Add Level 1 tests if available
        if test_level >= 1:
            try:
                level1_tests = unittest.TestLoader().loadTestsFromTestCase(Level1MockTests)
                suite.addTests(level1_tests)
            except Exception as e:
                print(f"Skipping Level 1 tests: {e}")
        
        # Add Level 2 tests if available
        if test_level >= 2:
            try:
                level2_tests = unittest.TestLoader().loadTestsFromTestCase(Level2IntegrationTests)
                suite.addTests(level2_tests)
            except Exception as e:
                print(f"Skipping Level 2 tests: {e}")
        
        return suite
    
    @classmethod
    def run_all_tests(cls):
        """Run all appropriate tests based on environment."""
        test_level = TestConfig.get_test_level()
        print(f"Running tests at level {test_level} (0=Standalone, 1=Mock, 2=Integration)")
        
        suite = cls.get_test_suite()
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result

if __name__ == "__main__":
    # Run unified test suite
    result = TestSuiteOrchestrator.run_all_tests()
    
    # Print summary
    print("\n" + "="*60)
    print("UNIFIED TEST SUITE SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)