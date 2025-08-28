import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from system import logging_utils
import requests
import urllib3
urllib3.disable_warnings()

# get token from SIEM > admin > authorized service

TOKEN_default = "677f60e2-3d58-4275-a1f0-c13d1975fdbe"
request_header_default = {
    'SEC': TOKEN_default,
    'Version': '20.0',
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Connection': 'close'
}

Qradar_address_default = "192.168.153.123"
search_id_default = "6c1b5627-e9f1-45a9-9040-7bab65a6463b"

### Delete a search in Qradar using the search_id
def delete_searches_Qradar(Qradar_address = Qradar_address_default, search_id = search_id_default, request_header = request_header_default):

    # Construct the request URI for the Qradar API
    request_URI = "https://" + Qradar_address + "/api/ariel/searches/" + search_id

    try:
        ## Make a DELETE request to the Qradar API
        delete_request_ariel_searches = requests.delete(request_URI, headers = request_header, verify=False)
        # Parse the JSON responses
        delete_response_ariel_searches = delete_request_ariel_searches.json()
        logging_utils.run_log("INFO", "10. DELETE Request sent to Qradar: deleting finished ariel searches -- Response Code:" + str(delete_request_ariel_searches))
        
    except:
        logging_utils.run_log("ERROR", "failed to send DELETE Request to Qradar")
        return

    # Return the response indicating the deletion result
    try:
        status = delete_response_ariel_searches["status"]
        logging_utils.run_log("INFO", "11. Response received from Qradar: deleted search -- status:" + str(status))
        
    except:
        logging_utils.run_log("ERROR", "Response received from Qradar: error message -- body:" + str(delete_response_ariel_searches))
        return

if __name__ == "__main__":
    deleted_searches = delete_searches_Qradar(Qradar_address = Qradar_address_default, search_id = search_id_default, request_header = request_header_default)
    print(deleted_searches)