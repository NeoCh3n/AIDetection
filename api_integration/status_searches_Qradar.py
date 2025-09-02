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

# get uri from
# https://10.33.232.84/api_doc?version=20.0&api=%2Fariel%2Fsearches%2F%7Bsearch_id%7D&method=GET
Qradar_address_default = "192.168.153.123"
search_id_default = "6c1b5627-e9f1-45a9-9040-7bab65a6463b"

def status_searches_Qradar(Qradar_address=Qradar_address_default, search_id=search_id_default,
                           request_header=request_header_default, timeout: int = 30):
    """Retrieves the status of a search in Qradar using the search_id"""
    # Construct the request URI for the Qradar API
    request_URI = f"https://{Qradar_address}/api/ariel/searches/" + search_id
    try:
        # Make a GET request to the Qradar API
        get_request_ariel_searches = requests.get(
            request_URI, headers=request_header, verify=False, timeout=timeout
        )
        # Parse the JSON response
        get_response_ariel_searches = get_request_ariel_searches.json()
        logging_utils.run_log("INFO", "5. GET Request sent to Qradar: getting ariel searches status information -- Response Code: " + str(get_request_ariel_searches))
    except requests.Timeout as e:
        logging_utils.run_log("ERROR", f"QRadar status check timeout after {timeout}s: {e}")
        return None
    except Exception as e:
        logging_utils.run_log("ERROR", f"Failed to send GET Request to Qradar: {e}")
        return None
    # return information from the response
    try:
        status = str(get_response_ariel_searches['status'])
        progress = str(get_response_ariel_searches['progress'])
        query_execution_time = str(get_response_ariel_searches['query_execution_time']/1000)
        logging_utils.run_log("INFO", "6. Response received from Qradar: retrieved ariel searches status information -- status:" + status + ", progress:" + progress + "%," + "query_execution_time:" + query_execution_time)
        return(get_response_ariel_searches)
    except Exception:
        logging_utils.run_log("ERROR", "Response received from Qradar: error message -- body:" + str(get_response_ariel_searches))
        return None

if __name__ == "__main__":
    status = status_searches_Qradar(Qradar_address_default, search_id_default, request_header_default)
    print(status)
