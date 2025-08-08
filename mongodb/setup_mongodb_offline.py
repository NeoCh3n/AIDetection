#!/usr/bin/env python3
"""
Updated MongoDB Offline Setup and Initialization Script
Compatible with RHEL 7.9 and macOS ARM

This updated script integrates with:
1. Unified MongoDB connection utility (mongodb_connection.py)
2. time_utils.py for consistent timestamp processing
3. Detection-only pipeline architecture
4. 30-minute sliding windows with 15-minute queries
"""

import subprocess
import sys
import os
import platform
from datetime import datetime, timedelta
import json
import numpy as np

# Add required paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared_utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'system'))

try:
    import pymongo
    from pymongo import MongoClient, IndexModel
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pymongo"])
    import pymongo
    from pymongo import MongoClient, IndexModel
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

class MongoDBOfflineSetup:
    def __init__(self, db_name="qradar_detection", host="localhost", port=27017):
        self.db_name = db_name
        self.host = host
        self.port = port
        self.connection_string = f"mongodb://{host}:{port}/"
        self.client = None
        self.db = None
        self.platform = platform.system()
        self.distro = self._get_linux_distro()
        
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
        print("🔍 Checking MongoDB installation...")
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
                    print(f"⏳ Retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                else:
                    print("Failed to connect to MongoDB")
                    return False
    
    def setup_collections(self):
        """Setup collections for detection-only pipeline with time_utils integration"""
        print("Setting up detection collections...")
        
        # Collections for detection-only mode with new schema
        collections = {
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
        
        for collection_name, indexes in collections.items():
            collection = self.db[collection_name]
            
            # Drop existing indexes (except _id)
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
        
        # Import time_utils for integration
        try:
            from time_utils import parse_qradar_timestamp, get_window_id, get_window_start_end
        except ImportError:
            print("Warning: time_utils not available, using manual calculation")
            def get_window_start_end(timestamp, window_size=30):
                minutes = timestamp.minute
                window_start = timestamp.replace(minute=(minutes // 30) * 30, second=0, microsecond=0)
                window_end = window_start + timedelta(minutes=30)
                return window_start, window_end
            
            def get_window_id(timestamp):
                start, _ = get_window_start_end(timestamp)
                return start.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Real rule IDs from AQLjsonResult.json
        real_rule_ids = [100227, 100221, 100272, 100277, 100101, 100216, 100215, 100225, 100218, 100265]
        real_counts = [211656, 211656, 210870, 210838, 210776, 210774, 6561, 6561, 6561, 6561]
        
        base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        
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
                "model_version": "ransomware_detector_v1.0",
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
            print(f"Sample window_id: {doc.get('_id')}")
            print(f"   Time window: {doc.get('window_start')} -> {doc.get('window_end')}")
            print(f"   Total triggers: {doc.get('total_triggers')}")
            print(f"   Window sequence: {doc.get('window_sequence')}")
        
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
                "total_rules": len(rule_ids),
                "rule_ids": rule_ids,
                "source": "AQLjsonResult.json",
                "description": "Dynamic rule extraction from QRadar AQL results"
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
    
    def run_setup(self):
        """Run the complete setup process"""
        print("MongoDB Offline Setup Starting...")
        print("=" * 50)
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
        
        # Setup collections
        self.setup_collections()
        
        # Insert sample data
        self.insert_sample_data()
        
        # Verify setup
        self.verify_setup()
        
        # Create config file
        self.create_config_file()
        
        print("\n" + "=" * 50)
        print("Updated MongoDB Detection Setup Complete!")
        print(f"Database: {self.db_name}")
        print(f"Collection: qradar_sliding_windows")
        print(f"Mode: Detection-only with 30-min sliding windows")
        print(f"Timezone: HKT (Asia/Hong_Kong)")
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