from pymongo import MongoClient
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'system'))
import run_log

#### MongoDB configuration for QRadar rule trigger ML pipeline (offline deployment)
CONNECTION_STRING_default = "mongodb://localhost:27017/"  # # Local MongoDB connection for offline use
NAME_DB_default = "qradar_ml"  # # Database for QRadar rule trigger ML pipeline

def get_database(CONNECTION_STRING_DB = CONNECTION_STRING_default, NAME_DB = NAME_DB_default):
    """
    Connect to MongoDB for QRadar rule trigger ML pipeline
    # # Returns database connection for offline deployment
    """
    try:
        # # Create connection using MongoClient for QRadar ML pipeline
        client_DB = MongoClient(CONNECTION_STRING_DB)
        # # Return pymongo.database.Database object for rule trigger storage and ML processing
        return client_DB[NAME_DB]

    except Exception as e:
        run_log.run_log("ERROR", "Failed to connect to MongoDB. Message: " + str(e))
        return

if __name__ == "__main__":
    # # Test MongoDB connection for QRadar ML pipeline
    DB = get_database(CONNECTION_STRING_default, NAME_DB_default)
    print(DB)
    print(f"Connected to QRadar ML database: {NAME_DB_default}")