import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from system import logging_utils
import requests
from urllib.parse import urlencode
import urllib3
urllib3.disable_warnings()

# get token from SIEM > admin > authorized service
TOKEN_default = "677f60e2-3d58-4275-a1f0-c13d1975fdbe"
request_header_default = {
    'SEC': TOKEN_default,
    'Version': '20.0',
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Connection': 'Close'
}

Qradar_address_default = "192.168.153.123"

AQL_default = """select "qidEventId" as 'Event ID',"sysmon_hostname" as 'sysmon_hostname (custom)',"deviceTime" as 'Log Source Time',"startTime" as 'Start Time',"Process Path" as 'Process Path (custom)',"sourceIP" as 'Source IP',"destinationIP" as 'Destination IP',"destinationPort" as 'Destination Port' from events where ( "qidEventId"='3' AND "sysmon_hostname"='DESKTOP-64-EDR' ) order by "startTime" desc start '2024-11-05 09:00' stop '2024-11-05 18:00'"""

#### Creates a search in Qradar using the specified parameters
def create_searches_Qradar(qradar_address=Qradar_address_default, AQL=AQL_default,
                           request_header=request_header_default, timeout: int = 60):
    # Construct the request URI for the Qradar API
    base_uri = "https://" + qradar_address + "/api/ariel/searches"

    try:
        # Make a POST request to the Qradar API
        # Use params so requests handles URL encoding safely
        post_request_ariel_searches = requests.post(
            base_uri,
            headers=request_header,
            params={"query_expression": AQL},
            verify=False,
            timeout=timeout,
        )
        # Parse the JSON response
        post_response_ariel_searches = post_request_ariel_searches.json()
        logging_utils.run_log("INFO", "2. POST Request sent to Qradar: ariel searches with AQL -- Response Code:" + str(post_request_ariel_searches))
    except requests.Timeout as e:
        logging_utils.run_log("ERROR", f"QRadar create_search timeout after {timeout}s: {e}")
        return None
    except Exception as e:
        logging_utils.run_log("ERROR", f"Failed to send POST Request to Qradar: {e}")
        return None

    try:
        # Return search_id from the response
        ariel_search_id = post_response_ariel_searches["search_id"]
        logging_utils.run_log("INFO", "3. Response received from Qradar: created ariel searches -- ariel_search_id:" + str(ariel_search_id))
        return post_response_ariel_searches
    except Exception:
        logging_utils.run_log("ERROR", "Response received from Qradar: error message ~ body:" + str(post_response_ariel_searches))
        return None

if __name__ == "__main__":
    result = create_searches_Qradar(qradar_address = Qradar_address_default, AQL = AQL_default, request_header = request_header_default)
    if result and "search_id" in result:
        search_id = result["search_id"]
        print(search_id)
    else:
        print("Error: Failed to create search or no search_id returned")
