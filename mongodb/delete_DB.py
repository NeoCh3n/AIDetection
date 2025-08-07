# Delete real-time QRadar data older than 7 days
# Automated 7-day retention for real-time storage only
import get_DB
from datetime import datetime, timedelta
import run_log

# MongoDB configuration for real-time data
CONNECTION_STRING = "mongodb://localhost:27017/"  # # Local MongoDB connection
DATABASE_NAME = "qradar_ml"  # # Database for real-time data
COLLECTION_NAME = "qradar_realtime"  # # Collection for 30-min real-time data

def delete_old_realtime_data(retention_days: int = 7):
    """
    # # Delete real-time data older than 7 days
    # # Training data handled separately in CSV files
    """
    try:
        # Calculate cutoff date (7 days ago from current time)
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        # Query to find documents older than retention period
        query = { 'time_window': { '$lt': cutoff_date } }
        
        # Connect to database
        DB = get_DB.get_database(CONNECTION_STRING, DATABASE_NAME)
        collection = DB[COLLECTION_NAME]
        
        # Delete old documents
        deleted_data = collection.delete_many(query)
        
        if deleted_data.deleted_count > 0:
            run_log.run_log("INFO", f"Deleted {deleted_data.deleted_count} old real-time records (older than {retention_days} days)")
            return deleted_data.deleted_count
        else:
            run_log.run_log("INFO", f"No old real-time records found to delete (older than {retention_days} days)")
            return 0
            
    except Exception as e:
        run_log.run_log("ERROR", f"Failed to delete old real-time data: {str(e)}")
        return 0

if __name__ == "__main__":
    deleted_count = delete_old_realtime_data(7)  # # Run 7-day cleanup
    print(f"7-day real-time cleanup completed: {deleted_count} records deleted")