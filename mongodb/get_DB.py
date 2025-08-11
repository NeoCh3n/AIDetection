from pymongo import MongoClient
import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from system import run_log

def get_database(config_path=None):
    """
    Connect to MongoDB using configuration from mongodb_config.json
    Returns database connection for ransomware detection pipeline
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'mongodb_config.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        mongo_config = config['mongodb']
        client_DB = MongoClient(mongo_config['connection_string'])
        return client_DB[mongo_config['db_name']]

    except Exception as e:
        run_log.run_log("ERROR", f"Failed to connect to MongoDB: {str(e)}")
        return None

if __name__ == "__main__":
    # # Test MongoDB connection for QRadar ML pipeline
    DB = get_database()
    if DB:
        print(DB)
        print(f"Connected to database: {DB.name}")
    else:
        print("Failed to connect to MongoDB")