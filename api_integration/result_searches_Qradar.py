import run_log
import requests
import urllib3
urllib3.disable_warnings()

# get token from SIEM > admin > authorized service
TOKEN_default = "677f60e2-3d58-4275-a1f0-c13d1975fdbe"

request_header_default={
    'SEC':TOKEN_default,
    'Version': '20.0',
    'Accept':'application/json',
    'Content-Type':'application/json',
    'Connection' : 'close'
}

Qradar_address_default = "192.168.153.123"
search_id_default = "6c1b5627-e9f1-45a9-9040-7bab65a6463b"

#### Retrieves the results of a search in Qradar using the search id
def result_searches_Qradar(Qradar_address = Qradar_address_default , search_id = search_id_default , request_header = request_header_default ):
    ## Construct the request URI for the Qradar API
    request_URI = "https://" + Qradar_address + "/api/ariel/searches/" + search_id + "/results"
    try:
        ## Make a GET request to the Qradar API
        get_request_ariel_searches_results = requests.get(request_URI, headers = request_header, verify =False)
        ## Parse the JSON response
        get_response_ariel_searches_results = get_request_ariel_searches_results.json()
        run_log.run_log("INFO" , "7. GET Request sent to Qradar: getting ariel searches results -- Response Code:" + str(get_request_ariel_searches_results))
    except:
        run_log.run_log("ERROR" , "Failed to send GET Request to Qradar")
        return
    ## return the results
    try:
        record_count = len(get_response_ariel_searches_results['events'])
        run_log.run_log("INFO" , "8. Response received from Qradar: downloaded ariel search results -- record_count:" + str(record_count) )
        return(get_response_ariel_searches_results)
    except:
        run_log.run_log("ERROR", "Response received from Qradar: error message -- body:" + str(get_response_ariel_searches_results) )
        return

#### default
if __name__ == "__main__":
    get_response_ariel_searches_results = result_searches_Qradar(Qradar_address = Qradar_address_default , search_id = search_id_default , request_header = request_header_default )
    try:
        print("Number of items:",len(get_response_ariel_searches_results["events"]))
    except:
        print(get_response_ariel_searches_results)
