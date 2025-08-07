# Get the database using the method we defined in pymongo_test_insert file
import get_DB
import run_log

from pandas import DataFrame

#### default parameters
CONNECTION_STRING_default = "mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.2.12"

NAME_default = "SIEM"

COLLECTION_default = "NetworkConnection"

QUERY_default = { }

#### Query the specified MongoDB collection and returns the data
def query_database(CONNECTION_STRING_DB = CONNECTION_STRING_default, NAME_DB = NAME_default, NAME_COLLECTION = COLLECTION_default, QUERY = QUERY_default):
    try:
        # Get the database using the collection string and database name
        DB = get_DB.get_database(CONNECTION_STRING_DB , NAME_DB)
        # Select the specified collection
        collection_name = DB[NAME_COLLECTION]
        # Execute the query to find all documents in the collection
        data = collection_name.find(QUERY)
        run_log.run_log( "INFO" , "12. Finished query data from MongoDB")
        return (data)
    except Exception as e:
        run_log.run_log( "ERROR" , "Failed to query data from MongoDB. Message: " + str(e) )

if __name__ == "__main__":
    data = query_database(CONNECTION_STRING_DB = CONNECTION_STRING_default , NAME_DB = NAME_default, NAME_COLLECTION = COLLECTION_default)
    for each in data:
        print("_id:",each['_id'])
        print("Date:",each['Date'])
        print("Time_interval:",each['Time_interval'])
        print("Count:",len(each['events']))
