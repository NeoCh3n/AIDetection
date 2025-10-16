#!/usr/bin/env python3
"""
Updated MongoDB Offline Setup and Initialization Script
Compatible with RHEL 7.9 and macOS ARM

COMPLETE MONGODB RULE SETUP PLAN
================================

ARCHITECTURE ALIGNMENT: Unified Processing Pipeline
--------------------------------------------------
This setup follows the unified data processing pipeline:
- Detection-only mode with 30-minute sliding windows
- 1128 production rules as fixed baseline
- Shared utilities: mongodb_connection.py, time_utils.py, rule_manager.py
- 15-minute query frequency with 30-minute aggregation windows

PHASE 1: RULE MANAGEMENT & VALIDATION
------------------------------------
1. Rule Discovery via rule_manager.py:
   - Fixed 1128 production rules (confirmed baseline)
   - Uses rule_manager.py for centralized rule ID management
   - Maps to Qradar_rule/ directory structure
   - Validates rule count consistency across environments

2. Configuration Validation:
   - mongodb_config.json contains 1128 total_rules
   - rule_mapping.json aligns with production baseline
   - UAT-to-Production mapping enabled via shared_utils/

PHASE 2: UNIFIED PIPELINE SETUP
------------------------------
3. MongoDB Integration:
   - Uses mongodb_connection.py (centralized connection utility)
   - Creates detection collections with time_utils integration
   - Sets up 30-minute window aggregation schema
   - Configures proper indexes for sliding window queries

4. Time Window Configuration:
   - 30-minute sliding windows (aligned with feature_aggregator.py)
   - 15-minute query frequency (detection mode)
   - HKT timezone handling via time_utils.py
   - Window ID generation consistent across pipeline

PHASE 3: PIPELINE VALIDATION
---------------------------
5. End-to-End Verification:
   - Test mongodb_connection.py integration
   - Validate time_utils window calculation
   - Confirm rule_manager.py rule mapping
   - Test 15-minute query → 30-minute aggregation pipeline

EXECUTION ORDER (Unified Pipeline):
----------------------------------
cd /Users/chaoyanchen/Desktop/AIDetection4Ransomware

# Step 1: Validate rule setup (1128 rules)
python shared_utils/rule_manager.py validate --count

# Step 2: Run complete MongoDB setup
python mongodb/setup_mongodb_offline.py

# Step 3: Test unified connection
python -c "from mongodb_connection import MongoDBConnection; print('✓ Connection OK')"

# Step 4: Validate pipeline integration
python shared_utils/time_utils.py test-window-calculation

PIPELINE INTEGRATION POINTS:
---------------------------
- data_loader.py → Uses mongodb_connection.py for MongoDB access
- feature_aggregator.py → Uses time_utils for window calculations
- feature_generator.py → Uses rule_manager.py for rule ID mapping
- model_predictor.py → Writes to detection_results collection
- All modules use unified configuration from mongodb_config.json

EXPECTED LOCATIONS:
------------------
- Rule definitions: Qradar_rule/ (managed by rule_manager.py)
- MongoDB config: mongodb/mongodb_config.json
- Rule mapping: shared_utils/rule_mapping.json
- UAT mapping: shared_utils/uat_to_prod_mapping.csv
- MongoDB collections: Created via mongodb_connection.py
- Time utilities: shared_utils/time_utils.py

This setup ensures zero training-serving skew by using identical rule baseline (1128 rules) across all environments.
"""

import subprocess
import sys
import os
import platform
from datetime import datetime, timedelta
import json
import numpy as np

# Add required paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared_utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'system'))

try:
    from pymongo import MongoClient, IndexModel
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pymongo"])
    from pymongo import MongoClient, IndexModel
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

class MongoDBOfflineSetup:
    def __init__(self, config_path=None):
        # Load configuration from mongodb_config.json
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'mongodb_config.json')
        
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # Create default config if not exists
            self.config = self._create_default_config()
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        
        self.mongo_config = self.config['mongodb']
        self.pipeline_config = self.config['pipeline']
        self.collections_config = self.config['collections']
        
        self.db_name = self.mongo_config['db_name']
        self.host = self.mongo_config['host']
        self.port = self.mongo_config['port']
        self.connection_string = self.mongo_config['connection_string']
        
        self.client = None
        self.db = None
        self.platform = platform.system()
        self.distro = self._get_linux_distro()
        
    def _create_default_config(self):
        """Create default configuration for MongoDB setup with UAT-Production mapping"""
        return {
            "mongodb": {
                "host": "localhost",
                "port": 27017,
                "db_name": "qradar_detection",
                "connection_string": "mongodb://localhost:27017/"
            },
            "pipeline": {
                "mode": "detection_only",
                "query_frequency_minutes": 15,
                "window_size": "30min",
                "timezone": "HKT",
                "retention_days": 7
            },
            "collections": {
                "detection_windows": "qradar_sliding_windows",
                "detection_results": "detection_results"
            },
            "data_schema": {
                "detection_windows": {
                    "_id": "string",
                    "window_start": "datetime",
                    "window_end": "datetime",
                    "query_time": "datetime",
                    "feature_vector": "object",
                    "rule_counts": "object",
                    "host_triggers": "object",
                    "total_triggers": "integer",
                    "total_rules_triggered": "integer"
                }
            },
            "rule_mapping": {
                "total_rules": 1128,
                "comment": "Fixed production rule count - using production rule coordinates as baseline",
                "source": "Production_baseline",
                "description": "Using 1128 production rules as baseline for consistent feature vector dimensions across environments",
                "uat_mapping": {
                    "enabled": True,
                    "mapping_file": "shared_utils/uat_to_prod_mapping.csv",
                    "description": "UAT-to-Production rule ID mapping for consistent training"
                }
            }
        }
        
    def _get_linux_distro(self):
        """Detect Linux distribution"""
        if platform.system() == "Linux":
            try:
                with open('/etc/redhat-release', 'r') as f:
                    return f.read().strip()
            except:
                return "Linux"
        return None
        
    def check_mongodb_installation(self):
        """Check if MongoDB is installed and running - cross-platform"""
        print("Checking MongoDB installation...")
        print(f"   Platform: {self.platform}")
        if self.distro:
            print(f"   Distribution: {self.distro}")
        
        # Check for MongoDB installation
        mongodb_paths = []
        if self.platform == "Darwin":  # macOS
            mongodb_paths = ["/usr/local/bin/mongod", "/opt/homebrew/bin/mongod"]
        elif self.platform == "Linux":
            mongodb_paths = ["/usr/bin/mongod", "/usr/local/bin/mongod", "/opt/mongodb/bin/mongod"]
        
        mongod_found = False
        for path in mongodb_paths:
            if os.path.exists(path):
                print(f"mongod found at: {path}")
                mongod_found = True
                break
        
        if not mongod_found:
            print("mongod not found in standard locations")
            return False
        
        # Check if MongoDB is running using platform-specific methods
        return self._check_mongodb_running()
        
    def _check_mongodb_running(self):
        """Check if MongoDB is running - platform-specific"""
        try:
            # Platform-specific process checking
            if self.platform == "Darwin":  # macOS
                try:
                    result = subprocess.run(['pgrep', '-f', 'mongod'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        print("MongoDB is running")
                        return True
                except FileNotFoundError:
                    pass
                    
                # macOS alternative with lsof
                try:
                    result = subprocess.run(['lsof', '-i', f':{self.port}'], 
                                          capture_output=True, text=True)
                    if 'mongod' in result.stdout:
                        print("MongoDB is running on port", self.port)
                        return True
                except FileNotFoundError:
                    pass
                    
            elif self.platform == "Linux":
                # RHEL 7.9 compatible checks
                try:
                    # Check with systemctl (RHEL 7+)
                    result = subprocess.run(['systemctl', 'is-active', 'mongod'], 
                                          capture_output=True, text=True)
                    if result.stdout.strip() == "active":
                        print("MongoDB service is active")
                        return True
                except FileNotFoundError:
                    pass
                
                # Check with service command (RHEL 7)
                try:
                    result = subprocess.run(['service', 'mongod', 'status'], 
                                          capture_output=True, text=True)
                    if "running" in result.stdout.lower():
                        print("MongoDB service is running")
                        return True
                except FileNotFoundError:
                    pass
                
                # Check with ps
                try:
                    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
                    if 'mongod' in result.stdout:
                        print("MongoDB process found")
                        return True
                except FileNotFoundError:
                    pass
                
                # Check with netstat
                try:
                    result = subprocess.run(['netstat', '-tlnp'], capture_output=True, text=True)
                    if f':{self.port}' in result.stdout and 'mongod' in result.stdout:
                        print("MongoDB is listening on port", self.port)
                        return True
                except FileNotFoundError:
                    pass
        
        except Exception as e:
            print(f"Error checking MongoDB status: {e}")
        
        # Final check - try connecting
        try:
            client = MongoClient(self.connection_string, serverSelectionTimeoutMS=2000)
            client.admin.command('ping')
            print("MongoDB is accessible")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError):
            print("MongoDB is not accessible")
            return False
    
    def start_mongodb(self):
        """Attempt to start MongoDB locally - platform-specific"""
        print("Starting MongoDB...")
        
        # Platform-specific installation checks
        if self.platform == "Darwin":  # macOS
            return self._start_mongodb_macos()
        elif self.platform == "Linux":
            return self._start_mongodb_linux()
        else:
            print(f"Unsupported platform: {self.platform}")
            return False
            
    def _start_mongodb_macos(self):
        """Start MongoDB on macOS"""
        print("   Platform: macOS")
        
        # Check if mongod is available via Homebrew
        try:
            subprocess.run(['mongod', '--version'], capture_output=True, check=True)
            print("mongod found via Homebrew")
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("mongod not found")
            print("Please install MongoDB:")
            print("  brew tap mongodb/brew")
            print("  brew install mongodb-community")
            return False
        
        return self._start_mongodb_common()
        
    def _start_mongodb_linux(self):
        """Start MongoDB on RHEL 7.9"""
        print("   Platform: Linux")
        
        # RHEL 7.9 specific MongoDB installation check
        try:
            subprocess.run(['mongod', '--version'], capture_output=True, check=True)
            print("mongod found")
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("mongod not found")
            print("For RHEL 7.9, install MongoDB:")
            print("  1. Create /etc/yum.repos.d/mongodb-org-4.4.repo")
            print("  2. sudo yum install mongodb-org")
            print("  3. sudo systemctl start mongod")
            print("  4. sudo systemctl enable mongod")
            return False
            
        return self._start_mongodb_common()
        
    def _start_mongodb_common(self):
        """Common MongoDB startup logic"""
        try:
            # Create data directory if it doesn't exist
            data_dir = os.path.join(os.path.expanduser("~"), "mongodb_data")
            os.makedirs(data_dir, exist_ok=True)
            
            # Check if already running
            if self._check_mongodb_running():
                print("MongoDB is already running")
                return True
            
            # Platform-specific startup commands
            if self.platform == "Linux":
                # Try systemctl first (RHEL 7+)
                try:
                    result = subprocess.run(['sudo', 'systemctl', 'start', 'mongod'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        print("MongoDB started via systemctl")
                        return True
                except FileNotFoundError:
                    pass
                
                # Try service command
                try:
                    result = subprocess.run(['sudo', 'service', 'mongod', 'start'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        print("MongoDB started via service")
                        return True
                except FileNotFoundError:
                    pass
            
            # Manual startup
            cmd = [
                'mongod',
                '--dbpath', data_dir,
                '--port', str(self.port),
                '--bind_ip', '127.0.0.1',
                '--fork', '--logpath', os.path.join(data_dir, 'mongod.log')
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("MongoDB started successfully")
                return True
            else:
                print("Failed to start MongoDB:", result.stderr)
                return False
                
        except Exception as e:
            print(f"Error starting MongoDB: {e}")
            return False
    
    def connect_to_mongodb(self):
        """Connect to MongoDB with retry logic"""
        print("Connecting to MongoDB...")
        
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.client = MongoClient(
                    self.connection_string,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=10000
                )
                # Test connection
                self.client.admin.command('ping')
                self.db = self.client[self.db_name]
                print("Connected to MongoDB")
                return True
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                else:
                    print("Failed to connect to MongoDB")
                    return False
    
    def setup_collections(self, force_recreate=False):
        """Setup collections for detection-only pipeline with time_utils integration"""
        print("Setting up detection collections...")
        
        # Collections for detection-only mode with new schema
        collections_dict = {
            'qradar_sliding_windows': [
                IndexModel([("window_start", -1), ("window_end", -1)]),
                IndexModel([("query_time", -1)]),
                IndexModel([("window_sequence", 1)]),
                IndexModel([("metadata.query_id", 1)]),
                IndexModel([("host_triggers", 1)]),
                IndexModel([("total_triggers", -1)]),
                IndexModel([("window_start", 1), ("window_end", 1)]),
            ],
            'detection_results': [
                IndexModel([("timestamp", -1)]),
                IndexModel([("window_id", 1)]),
                IndexModel([("prediction", 1)]),
                IndexModel([("confidence", -1)]),
            ]
        }
        
        if self.db is None:
            raise RuntimeError("Database not connected")
            
        for collection_name, indexes in collections_dict.items():
            collection = self.db[collection_name]
            
            # Check if collection exists and has data
            doc_count = collection.count_documents({})
            
            if doc_count > 0 and not force_recreate:
                print(f"Collection {collection_name} has {doc_count} documents - preserving existing data and indexes")
                # Verify required indexes exist, create missing ones
                existing_indexes = [idx['name'] for idx in collection.list_indexes()]
                missing_indexes = []
                
                for new_idx in indexes or []:
                    idx_name = new_idx.document.get('name', f"idx_{collection_name}")
                    if idx_name not in existing_indexes:
                        missing_indexes.append(new_idx)
                
                if missing_indexes:
                    collection.create_indexes(missing_indexes)
                    print(f"Created {len(missing_indexes)} missing indexes for {collection_name}")
                else:
                    print(f"Verified {collection_name} has all required indexes")
            else:
                # Safe to drop and recreate indexes for empty collections or forced recreation
                if force_recreate and doc_count > 0:
                    print(f"Force recreating indexes for {collection_name} with {doc_count} documents")
                
                if doc_count == 0:
                    print(f"Setting up fresh collection: {collection_name}")
                
                # Drop existing indexes (except _id) only for empty collections or forced recreation
                if force_recreate or doc_count == 0:
                    collection.drop_indexes()
                
                # Create new indexes
                if indexes:
                    collection.create_indexes(indexes)
                    print(f"Created indexes for {collection_name}")
                else:
                    print(f"Verified {collection_name} exists")
    
    def insert_sample_data(self):
        """Insert sample detection data using real AQL schema and time_utils"""
        print("Inserting sample detection data...")
        
        if self.db is None:
            raise RuntimeError("Database connection not established. Call connect_to_mongodb() first.")
        
        # Define local time utility functions to avoid import issues
        def get_window_start_end(timestamp):
            """Calculate window start and end times"""
            minutes = timestamp.minute
            window_start = timestamp.replace(minute=(minutes // 30) * 30, second=0, microsecond=0)
            window_end = window_start + timedelta(minutes=30)
            return window_start, window_end
        
        def get_window_id(timestamp):
            """Generate window ID from timestamp"""
            start, _ = get_window_start_end(timestamp)
            return start.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Real rule IDs from AQLjsonResult.json
        real_rule_ids = [100227, 100221, 100272, 100277, 100101, 100216, 100215, 100225, 100218, 100265]
        real_counts = [211656, 211656, 210870, 210838, 210776, 210774, 6561, 6561, 6561, 6561]
        
        # Simulate 15-minute queries upserting 30-minute window documents
        # Align base time to nearest 15-minute mark
        base_time = datetime.now().replace(minute=(datetime.now().minute // 15) * 15, second=0, microsecond=0)
        collection = self.db['qradar_sliding_windows']
        unique_ids = set()
        
        for query_idx in range(96):  # 24h * (60/15)
            query_time_anchor = base_time - timedelta(minutes=15 * query_idx)
            # Window aligns to 30-min boundaries using time_utils
            window_start, window_end = get_window_start_end(query_time_anchor)
            window_id = get_window_id(query_time_anchor)
            unique_ids.add(window_id)
            
            # Create feature vector with realistic variation
            feature_vector = {}
            rule_counts = {}
            total_triggers = 0
            for rule_id, base_count in zip(real_rule_ids, real_counts):
                variation = np.random.uniform(0.8, 1.2)
                count = int(base_count * variation)
                rule_str = str(rule_id)
                feature_vector[rule_str] = count
                rule_counts[rule_str] = count
                total_triggers += count
            
            # Create host-level breakdown
            host_triggers = {
                "192.168.153.166": {
                    "total_triggers": int(total_triggers * 0.3),
                    "rules": {str(rule_id): int(count * 0.3) for rule_id, count in zip(real_rule_ids[:5], real_counts[:5])}
                },
                "DESKTOP-64-EDR": {
                    "total_triggers": int(total_triggers * 0.7),
                    "rules": {str(rule_id): int(count * 0.7) for rule_id, count in zip(real_rule_ids, real_counts)}
                }
            }
            
            window_sequence = (window_start.hour * 60 + window_start.minute) // 15 + 1
            detection_window = {
                "_id": window_id,
                "window_start": window_start,
                "window_end": window_end,
                "query_time": window_end + timedelta(seconds=15),
                "window_sequence": window_sequence,
                "feature_vector": feature_vector,
                "rule_counts": rule_counts,
                "host_triggers": host_triggers,
                "total_triggers": total_triggers,
                "total_rules_triggered": len(real_rule_ids),
                "metadata": {
                    "query_id": f"q{(window_end + timedelta(seconds=15)).strftime('%Y%m%d_%H%M%S')}",
                    "source": "qradar_aql",
                    "data_type": "detection",
                    "overlap_previous": True
                }
            }
            
            # Upsert per window_id (overwrite latest state for the 30-min window)
            collection.replace_one({'_id': window_id}, detection_window, upsert=True)
        
        print(f"Upserted sliding windows for {len(unique_ids)} unique 30-min windows")
        
        # Insert detection results
        detection_results = []
        for window_idx in range(48):
            event_time = base_time - timedelta(minutes=30 * window_idx)
            window_id = get_window_id(event_time)
            
            # Simulate realistic detection with low false positive rate
            is_anomaly = np.random.choice([0, 1], p=[0.97, 0.03])
            
            result = {
                "window_id": window_id,
                "timestamp": event_time,
                "prediction": int(is_anomaly),
                "confidence": float(np.random.uniform(0.85, 0.99) if is_anomaly else np.random.uniform(0.70, 0.85)),
                "model_version": "threat_detector_v1.0",
                "total_events": sum(real_counts),
                "unique_rules": len(real_rule_ids)
            }
            detection_results.append(result)
        
        if detection_results:
            collection = self.db['detection_results']
            result = collection.insert_many(detection_results)
            print(f"Inserted {len(result.inserted_ids)} detection results")
        
    def verify_setup(self):
        """Verify the setup is working correctly"""
        print("Verifying setup...")
        
        if self.db is None:
            print("Error: Database not connected")
            return False
        
        # Check collections
        collections = self.db.list_collection_names()
        print(f"Available collections: {collections}")
        
        # Check sample data
        collection = self.db['qradar_sliding_windows']
        count = collection.count_documents({})
        print(f"Detection windows: {count}")
        
        if count > 0:
            # Show first document
            doc = collection.find_one()
            if doc:
                print(f"Sample window_id: {doc.get('_id')}")
                print(f"   Time window: {doc.get('window_start')} -> {doc.get('window_end')}")
                print(f"   Total triggers: {doc.get('total_triggers')}")
                print(f"   Window sequence: {doc.get('window_sequence')}")
            else:
                print("No sample data available")
        else:
            print("No documents in collection")
        
        return True
    
    def create_config_file(self):
        """Create updated configuration file with detection-only schema"""
        # Load actual rule data from AQLjsonResult.json
        aql_file = os.path.join(os.path.dirname(__file__), '..', 'AQLjsonResult.json')
        rule_ids = []
        
        try:
            with open(aql_file, 'r') as f:
                aql_data = json.load(f)
                rule_ids = list(set(str(event['Custom Rule']) for event in aql_data.get('events', [])))
        except:
            # Fallback to known rules
            rule_ids = ["100227", "100221", "100272", "100277", "100101", "100216", "100215", "100225", "100218", "100265"]
        
        config = {
            "mongodb": {
                "host": self.host,
                "port": self.port,
                "db_name": self.db_name,
                "connection_string": self.connection_string
            },
            "pipeline": {
                "mode": "detection_only",
                "query_frequency_minutes": 15,
                "window_size": "30min",
                "timezone": "HKT",
                "retention_days": 7
            },
            "collections": {
                "detection_windows": "qradar_sliding_windows",
                "detection_results": "detection_results"
            },
            "data_schema": {
                "detection_windows": {
                    "_id": "string (window_id from time_utils)",
                    "window_start": "datetime (from time_utils)",
                    "window_end": "datetime (from time_utils)",
                    "query_time": "datetime",
                    "feature_vector": "object (rule_id -> count)",
                    "rule_counts": "object (rule_id -> count)",
                    "host_triggers": "object (hostname -> {total_triggers, rules})",
                    "total_triggers": "integer",
                    "total_rules_triggered": "integer"
                }
            },
            "rule_mapping": {
                "total_rules": 1128,
                "rule_ids": rule_ids,
                "source": "Production_baseline",
                "description": "Fixed 1128 production rules as baseline for consistent feature vectors across environments",
                "fixed_count": True,
                "uat_mapping": {
                    "enabled": True,
                    "mapping_file": "shared_utils/uat_to_prod_mapping.csv",
                    "description": "UAT-to-Production rule ID mapping for consistent training"
                }
            },
            "time_utils": {
                "timezone": "Asia/Hong_Kong",
                "window_size_minutes": 30,
                "tolerance_seconds": 5,
                "timestamp_format": "%b %d, %Y, %I:%M:%S %p"
            }
        }
        
        config_path = os.path.join(os.path.dirname(__file__), 'mongodb_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        print(f"Configuration saved to {config_path}")
    
    def run_setup(self, force_recreate=False):
        """Run the complete setup process"""
        print("MongoDB Offline Setup Starting...")
        print("-" * 50)
        print(f"   Target Platform: {self.platform}")
        if self.distro:
            print(f"   Distribution: {self.distro}")
        
        # Check MongoDB
        if not self.check_mongodb_installation():
            print("\nSetup Instructions:")
            if self.platform == "Darwin":
                print("1. Install MongoDB on macOS:")
                print("   brew tap mongodb/brew")
                print("   brew install mongodb-community")
                print("   brew services start mongodb-community")
            elif self.platform == "Linux":
                if "Red Hat" in str(self.distro) or "CentOS" in str(self.distro):
                    print("1. Install MongoDB on RHEL 7.9:")
                    print("   sudo vi /etc/yum.repos.d/mongodb-org-4.4.repo")
                    print("   # Add MongoDB repository:")
                    print("   [mongodb-org-4.4]")
                    print("   name=MongoDB Repository")
                    print("   baseurl=https://repo.mongodb.org/yum/redhat/$releasever/mongodb-org/4.4/x86_64/")
                    print("   gpgcheck=1")
                    print("   enabled=1")
                    print("   gpgkey=https://www.mongodb.org/static/pgp/server-4.4.asc")
                    print("   sudo yum install mongodb-org")
                    print("   sudo systemctl start mongod")
                    print("   sudo systemctl enable mongod")
                else:
                    print("1. Install MongoDB on Linux:")
                    print("   sudo apt install mongodb (Ubuntu/Debian)")
                    print("   sudo yum install mongodb-org (CentOS/RHEL)")
            print("2. Run this script again")
            return False
        
        # Connect to MongoDB
        if not self.connect_to_mongodb():
            return False
        
        if self.db is None:
            print("Error: Failed to establish database connection")
            return False
        
        # Setup collections with safety check
        self.setup_collections(force_recreate=force_recreate)
        
        # Insert sample data only for fresh setup or forced recreation
        collection_name = 'qradar_sliding_windows'
        try:
            collection = self.db[collection_name]
            doc_count = collection.count_documents({})
            if force_recreate or doc_count == 0:
                self.insert_sample_data()
            else:
                print("Skipping sample data insertion - collection already has data")
        except Exception as e:
            print(f"Error checking collection: {e}")
            return False
        
        # Verify setup
        self.verify_setup()
        
        # Update config file with current settings
        self.create_config_file()
        
        print("\n" + "-" * 50)
        print("Updated MongoDB Detection Setup Complete!")
        print(f"Database: {self.db_name}")
        print(f"Collection: qradar_sliding_windows")
        print(f"Mode: Detection-only with 30-min sliding windows")
        print(f"Timezone: HKT (Asia/Hong_Kong)")
        print(f"Force recreate: {'Yes' if force_recreate else 'No'}")
        print("Next steps:")
        print("   - Use mongodb_connection.py for unified MongoDB operations")
        print("   - Run tests/test_aql_insert.py to test time_utils integration")
        print("   - Use mongodb/insert_DB.py for production data processing")
        print("   - Check mongodb_config.json for configuration details")
        
        return True
    
    def cleanup(self):
        """Clean up MongoDB connection"""
        if self.client:
            self.client.close()


def main():
    """Main setup function"""
    setup = MongoDBOfflineSetup()
    try:
        setup.run_setup()
    finally:
        setup.cleanup()


if __name__ == "__main__":
    main()