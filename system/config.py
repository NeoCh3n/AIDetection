import os

#### change parameters here

#### Qradar configuration
Qradar_address_default = "192.168.153.123"

Qradar_token_default = "677f60e2-3d58-4275-a1f0-c13d1975fdbe"

request_header_default = {
    'SEC': Qradar_token_default,
    'Version': '20.0',
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Connection': 'Close'
}

##### MongoDB configuration for QRadar rule trigger ML pipeline (offline deployment)
# DB CONNECTION STRING - Updated for offline MongoDB deployment
MONGODB_CONNECTION_STRING = "mongodb://localhost:27017/"

# DB NAME - Updated for QRadar ML project
MONGODB_DB_NAME = "qradar_ml"

# COLLECTION NAMES - Updated for rule trigger storage and ML processing
# Main collection for 15-minute rule trigger data
MONGODB_COLLECTION_NAME = "qradar_rule_triggers_ml"
# Additional collections for the complete pipeline
MONGODB_ML_FEATURES_COLLECTION = "ml_features"
MONGODB_ANOMALY_RESULTS_COLLECTION = "anomaly_results"

# Timeframe configuration for 15-minute storage with 30-minute testing
MONGODB_STORAGE_BUCKET_MINUTES = 15
MONGODB_TESTING_WINDOW_MINUTES = 30

# Data retention configuration - 7 days for operational requirements
DATA_RETENTION_DAYS = 7

### Change AQL here
AQL_default = """select "qidEventId" as 'Event ID',"sysmon_hostname" as 'sysmon_hostname (custom)', "deviceTime" as 'Log Source Time',"startTime" as 'Start Time',"Process Path" as 'Process Path (custom)',"sourceIP" as 'Source IP',"destinationIP" as 'Destination IP',"destinationPort" as 'Destination Port' from events where ( "qidEventId"='3' AND "sysmon_hostname"='DESKTOP-64-EDR' ) order by "startTime" desc last 15 minutes"""

#################
# Parameters
# Frequency of fetching data from Qradar (every __ minutes)
# if you change fetch_data_frequency_default, please also edit the schedule task in crontab
fetch_data_frequency_default = 15

# Analyze time (every __ minutes)
# Please do not change
# if you change analyze_time_default, please train and apply a new model
analyze_duration_default = 30

# Maximum record count
maximum_record_count_default = 800000

model_path_default = os.path.join("Model", "svm_NetworkConnection_4D")
svm_model_path_default = os.path.join("Model", "svm_NetworkConnection_4D")
logreg_model_path_default = os.path.join("Model", "logreg_NetworkConnection_4D")
rf_model_path_default = os.path.join("Model", "rf_NetworkConnection_4D")
voting_model_path_default = os.path.join("Model", "voting_NetworkConnection_4D")

log_dir_path_default = os.path.join("running_log/")

log_destination_address_default = "192.168.153.123"
log_destination_port_default = 514

#### NEVER change the below!
# run the default to check the parameter
if __name__ == "__main__":
    print("Qradar_address_default:", Qradar_address_default)
    print("Qradar_token_default:", Qradar_token_default)
    print("request_header_default:", request_header_default)
    print("MONGODB_CONNECTION_STRING:", MONGODB_CONNECTION_STRING)
    print("MONGODB_DB_NAME:", MONGODB_DB_NAME)
    print("MONGODB_COLLECTION_NAME:", MONGODB_COLLECTION_NAME)
    print("AQL_default:", AQL_default)
    print("fetch_data_frequency_default:", fetch_data_frequency_default)
    print("analyze_duration_default:", analyze_duration_default)
    print("maximum_record_count_default:", maximum_record_count_default)
    print("model_path_default:", model_path_default)
    print("svm_model_path_default:", svm_model_path_default)
    print("logreg_model_path_default:", logreg_model_path_default)
    print("rf_model_path_default:", rf_model_path_default)
    print("voting_model_path_default:", voting_model_path_default)
    print("log_dir_path_default:", log_dir_path_default)
    print("log_destination_address_default:", log_destination_address_default)
    print("log_destination_port_default:", log_destination_port_default)