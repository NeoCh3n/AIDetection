import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import get_DB
from system import run_log

#### Configuration from mongodb_config.json
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'mongodb_config.json')

def load_config():
    """Load configuration from mongodb_config.json"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        run_log.run_log("ERROR", f"Failed to load config: {str(e)}")
        return None

# Load configuration
config = load_config()
if config:
    CONNECTION_STRING_default = config['mongodb']['connection_string']
    NAME_default = config['mongodb']['db_name']
    COLLECTION_default = config['collections']['detection_windows']
    QUERY_default = {}
else:
    # Fallback defaults
    CONNECTION_STRING_default = "mongodb://localhost:27017/"
    NAME_default = "qradar_detection"
    COLLECTION_default = "qradar_sliding_windows"
    QUERY_default = {}

#### Query the specified MongoDB collection and returns the data
def query_database(NAME_COLLECTION = COLLECTION_default, QUERY = QUERY_default):
    try:
        # Get the database using configuration
        DB = get_DB.get_database()
        if DB is None:
            run_log.run_log("ERROR", "Failed to connect to MongoDB")
            return []
            
        # Select the specified collection
        collection_name = DB[NAME_COLLECTION]
        # Execute the query to find all documents in the collection
        data = list(collection_name.find(QUERY))
        run_log.run_log("INFO", f"12. Finished query data from MongoDB: {len(data)} documents")
        return data
    except Exception as e:
        run_log.run_log("ERROR", f"Failed to query data from MongoDB. Message: {str(e)}")
        return []

if __name__ == "__main__":
    data = query_database(COLLECTION_default)
    for each in data:
        print("_id:",each.get('_id', 'N/A'))
        print("Data:",each)
