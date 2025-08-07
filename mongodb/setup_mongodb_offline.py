#!/usr/bin/env python3
"""
MongoDB Offline Setup and Initialization Script
Compatible with RHEL 7.9 and macOS ARM

This script sets up MongoDB for offline deployment with:
1. Cross-platform MongoDB installation check (RHEL 7.9 / macOS ARM)
2. Database initialization with proper collections
3. Index creation for optimal performance
4. Sample data insertion for testing
5. Configuration for offline use
"""

import subprocess
import sys
import os
import platform
from datetime import datetime, timedelta
import json
import numpy as np

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
    def __init__(self, db_name="qradar_ml", host="localhost", port=27017):
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
                print(f"✅ mongod found at: {path}")
                mongod_found = True
                break
        
        if not mongod_found:
            print("❌ mongod not found in standard locations")
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
                        print("✅ MongoDB is running")
                        return True
                except FileNotFoundError:
                    pass
                    
                # macOS alternative with lsof
                try:
                    result = subprocess.run(['lsof', '-i', f':{self.port}'], 
                                          capture_output=True, text=True)
                    if 'mongod' in result.stdout:
                        print("✅ MongoDB is running on port", self.port)
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
                        print("✅ MongoDB service is active")
                        return True
                except FileNotFoundError:
                    pass
                
                # Check with service command (RHEL 7)
                try:
                    result = subprocess.run(['service', 'mongod', 'status'], 
                                          capture_output=True, text=True)
                    if "running" in result.stdout.lower():
                        print("✅ MongoDB service is running")
                        return True
                except FileNotFoundError:
                    pass
                
                # Check with ps
                try:
                    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
                    if 'mongod' in result.stdout:
                        print("✅ MongoDB process found")
                        return True
                except FileNotFoundError:
                    pass
                
                # Check with netstat
                try:
                    result = subprocess.run(['netstat', '-tlnp'], capture_output=True, text=True)
                    if f':{self.port}' in result.stdout and 'mongod' in result.stdout:
                        print("✅ MongoDB is listening on port", self.port)
                        return True
                except FileNotFoundError:
                    pass
        
        except Exception as e:
            print(f"⚠️  Error checking MongoDB status: {e}")
        
        # Final check - try connecting
        try:
            client = MongoClient(self.connection_string, serverSelectionTimeoutMS=2000)
            client.admin.command('ping')
            print("✅ MongoDB is accessible")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError):
            print("❌ MongoDB is not accessible")
            return False
    
    def start_mongodb(self):
        """Attempt to start MongoDB locally - platform-specific"""
        print("🚀 Starting MongoDB...")
        
        # Platform-specific installation checks
        if self.platform == "Darwin":  # macOS
            return self._start_mongodb_macos()
        elif self.platform == "Linux":
            return self._start_mongodb_linux()
        else:
            print(f"❌ Unsupported platform: {self.platform}")
            return False
            
    def _start_mongodb_macos(self):
        """Start MongoDB on macOS"""
        print("   Platform: macOS")
        
        # Check if mongod is available via Homebrew
        try:
            subprocess.run(['mongod', '--version'], capture_output=True, check=True)
            print("✅ mongod found via Homebrew")
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("❌ mongod not found")
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
            print("✅ mongod found")
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("❌ mongod not found")
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
                print("✅ MongoDB is already running")
                return True
            
            # Platform-specific startup commands
            if self.platform == "Linux":
                # Try systemctl first (RHEL 7+)
                try:
                    result = subprocess.run(['sudo', 'systemctl', 'start', 'mongod'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        print("✅ MongoDB started via systemctl")
                        return True
                except FileNotFoundError:
                    pass
                
                # Try service command
                try:
                    result = subprocess.run(['sudo', 'service', 'mongod', 'start'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        print("✅ MongoDB started via service")
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
                print("✅ MongoDB started successfully")
                return True
            else:
                print("❌ Failed to start MongoDB:", result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Error starting MongoDB: {e}")
            return False
    
    def connect_to_mongodb(self):
        """Connect to MongoDB with retry logic"""
        print("🔗 Connecting to MongoDB...")
        
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
                print("✅ Connected to MongoDB")
                return True
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                print(f"⚠️  Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"⏳ Retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                else:
                    print("❌ Failed to connect to MongoDB")
                    return False
    
    def setup_collections(self):
        """Setup collections for unified data processing pipeline as per CLAUDE.md"""
        print("📊 Setting up collections for unified pipeline...")
        
        # Collections aligned with CLAUDE.md pipeline structure
        collections = {
            'qradar_events': [
                IndexModel([("timestamp", 1)]),
                IndexModel([("hostname", 1)]),
                IndexModel([("rule_id", 1)]),
                IndexModel([("hostname", 1), ("timestamp", 1)]),
                IndexModel([("timestamp", -1)]),
            ],
            'time_windows': [
                IndexModel([("window_id", 1)]),
                IndexModel([("hostname", 1)]),
                IndexModel([("start_time", 1)]),
                IndexModel([("end_time", 1)]),
                IndexModel([("hostname", 1), ("window_id", 1)]),
            ],
            'training_data': [
                IndexModel([("window_id", 1)]),
                IndexModel([("hostname", 1)]),
                IndexModel([("is_attack", 1)]),
                IndexModel([("timestamp", 1)]),
            ],
            'detection_results': [
                IndexModel([("timestamp", -1)]),
                IndexModel([("hostname", 1)]),
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
                print(f"✅ Created indexes for {collection_name}")
            else:
                print(f"✅ Verified {collection_name} exists")
    
    def insert_sample_data(self):
        """Insert sample data aligned with CLAUDE.md pipeline structure"""
        print("📥 Inserting sample data for unified pipeline...")
        
        # Sample rule IDs from QRadar (aligned with training data)
        sample_rule_ids = [100001, 100005, 100010, 100015, 100020, 100025, 100030, 100035, 100040, 100045]
        
        base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        # 1. Insert raw qradar_events (for detection mode)
        print("   Inserting qradar_events...")
        events = []
        for i in range(100):
            timestamp = base_time - timedelta(minutes=i*30)
            event = {
                'hostname': f'DESKTOP-{np.random.randint(10, 99)}-EDR',
                'rule_id': np.random.choice(sample_rule_ids),
                'timestamp': timestamp,
                'count': np.random.poisson(5) + 1,
                'source': 'qradar'
            }
            events.append(event)
        
        if events:
            collection = self.db['qradar_events']
            result = collection.insert_many(events)
            print(f"   ✅ Inserted {len(result.inserted_ids)} qradar_events")
        
        # 2. Insert time_windows (for training mode)
        print("   Inserting time_windows...")
        windows = []
        for i in range(20):
            start_time = base_time - timedelta(hours=i)
            end_time = start_time + timedelta(minutes=30)
            window = {
                'window_id': f"window_{i}",
                'hostname': f'DESKTOP-{np.random.randint(10, 99)}-EDR',
                'start_time': start_time,
                'end_time': end_time,
                'rule_counts': {str(np.random.choice(sample_rule_ids)): np.random.poisson(3) + 1 
                               for _ in range(np.random.randint(3, 8))},
                'total_events': np.random.poisson(50) + 10
            }
            windows.append(window)
        
        if windows:
            collection = self.db['time_windows']
            result = collection.insert_many(windows)
            print(f"   ✅ Inserted {len(result.inserted_ids)} time_windows")
        
        # 3. Insert training_data (labeled data for ML)
        print("   Inserting training_data...")
        training_docs = []
        for i in range(50):
            start_time = base_time - timedelta(hours=i*2)
            is_attack = i % 5 == 0  # 20% attack samples
            doc = {
                'window_id': f"training_{i}",
                'hostname': f'DESKTOP-{np.random.randint(10, 99)}-EDR',
                'timestamp': start_time,
                'feature_vector': [np.random.poisson(2) if not is_attack else np.random.poisson(10) 
                                  for _ in range(10)],  # Simplified 10-rule vector
                'is_attack': int(is_attack),
                'label': 'attack' if is_attack else 'normal'
            }
            training_docs.append(doc)
        
        if training_docs:
            collection = self.db['training_data']
            result = collection.insert_many(training_docs)
            print(f"   ✅ Inserted {len(result.inserted_ids)} training samples")
        
        # 4. Insert detection_results (model predictions)
        print("   Inserting detection_results...")
        results = []
        for i in range(25):
            timestamp = base_time - timedelta(hours=i)
            prediction = np.random.choice([0, 1], p=[0.85, 0.15])  # 15% anomaly rate
            result = {
                'timestamp': timestamp,
                'hostname': f'DESKTOP-{np.random.randint(10, 99)}-EDR',
                'prediction': prediction,
                'confidence': np.random.uniform(0.7, 0.99),
                'window_id': f"result_{i}",
                'model_version': 'ransomware_detector_v1.0'
            }
            results.append(result)
        
        if results:
            collection = self.db['detection_results']
            result = collection.insert_many(results)
            print(f"   ✅ Inserted {len(result.inserted_ids)} detection results")
        
    def verify_setup(self):
        """Verify the setup is working correctly"""
        print("✅ Verifying setup...")
        
        # Check collections
        collections = self.db.list_collection_names()
        print(f"📋 Available collections: {collections}")
        
        # Check sample data
        collection = self.db['qradar_rule_triggers']
        count = collection.count_documents({})
        print(f"📊 Sample data count: {count}")
        
        if count > 0:
            # Show first document
            doc = collection.find_one()
            print(f"📄 Sample document: {doc['_id']}")
            print(f"   Rules triggered: {doc['unique_rules']}")
            print(f"   Total events: {doc['total_triggers']}")
        
        return True
    
    def create_config_file(self):
        """Create configuration file aligned with CLAUDE.md unified pipeline"""
        config = {
            "mongodb": {
                "host": self.host,
                "port": self.port,
                "db_name": self.db_name,
                "connection_string": self.connection_string
            },
            "pipeline": {
                "window_size": "30min",
                "timezone": "local"
            },
            "collections": {
                "qradar_events": "qradar_events",
                "time_windows": "time_windows",
                "training_data": "training_data",
                "detection_results": "detection_results"
            },
            "data_schema": {
                "qradar_events": {
                    "hostname": "string",
                    "rule_id": "integer",
                    "timestamp": "datetime",
                    "count": "integer"
                },
                "time_windows": {
                    "window_id": "string",
                    "hostname": "string",
                    "start_time": "datetime",
                    "end_time": "datetime",
                    "rule_counts": "dict"
                },
                "training_data": {
                    "window_id": "string",
                    "hostname": "string",
                    "feature_vector": "list",
                    "is_attack": "integer",
                    "timestamp": "datetime"
                }
            }
        }
        
        config_path = os.path.join(os.path.dirname(__file__), 'mongodb_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        print(f"✅ Configuration saved to {config_path}")
    
    def run_setup(self):
        """Run the complete setup process"""
        print("🚀 MongoDB Offline Setup Starting...")
        print("=" * 50)
        print(f"   Target Platform: {self.platform}")
        if self.distro:
            print(f"   Distribution: {self.distro}")
        
        # Check MongoDB
        if not self.check_mongodb_installation():
            print("\n📋 Setup Instructions:")
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
        print("✅ MongoDB Offline Setup Complete!")
        print(f"🎯 Database: {self.db_name}")
        print(f"🔗 Connection: {self.connection_string}")
        print("📝 Next steps:")
        print("   - Use data_loader.py with mode='detect' for MongoDB data loading")
        print("   - Run pipeline/data_loader.py to test data ingestion")
        print("   - Use shared_utils/config.py for configuration management")
        
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