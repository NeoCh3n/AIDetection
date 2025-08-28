import os

#### change parameters here

#### Qradar configuration
# Prefer environment variables; fall back to sensible defaults.
# Never commit real tokens into source control.
Qradar_address_default = os.getenv("QRADAR_ADDRESS", "192.168.153.123")
Qradar_token_default = os.getenv("QRADAR_API_TOKEN", "REPLACE_WITH_TOKEN")

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

# Use lowercase 'model' directory to match repository layout
model_path_default = os.path.join("model", "svm_NetworkConnection_4D")
svm_model_path_default = os.path.join("model", "svm_NetworkConnection_4D")
logreg_model_path_default = os.path.join("model", "logreg_NetworkConnection_4D")
rf_model_path_default = os.path.join("model", "rf_NetworkConnection_4D")
voting_model_path_default = os.path.join("model", "voting_NetworkConnection_4D")

log_dir_path_default = os.path.join("running_log/")

log_destination_address_default = os.getenv("SYSLOG_ADDRESS", "192.168.153.123")
log_destination_port_default = int(os.getenv("SYSLOG_PORT", "514"))

# Syslog header configuration (project name: AIR; default tag: AIR-RF)
SYSLOG_HEADER_BASE = os.getenv("SYSLOG_HEADER_BASE", "AIR")
SYSLOG_HEADER_ML = os.getenv("SYSLOG_HEADER_ML", f"{SYSLOG_HEADER_BASE}-RF")
SYSLOG_HEADER_LOG = os.getenv("SYSLOG_HEADER_LOG", f"{SYSLOG_HEADER_BASE}-RF")

#### Configuration dictionary for pipeline modules
def get_config():
    """
    Returns centralized configuration dictionary for pipeline components.
    Used by feature_generator.py and other modules.
    """
    return {
        'rule_manager': {
            'mode': 'file',
            'rule_file_path': os.path.join('Qradar_rule', 'production_rules.txt'),
            'uat_rule_file_path': os.path.join('Qradar_rule', 'uat_rules.txt'),
            'mapping_file_path': os.path.join('Qradar_rule', 'uat_to_prod_mapping.json'),
            'environment': os.getenv('ENVIRONMENT', 'prod')
        },
        'feature_engineering': {
            'expected_rules': 2898,
            'window_size_minutes': 30,
            'log_transform': True,
            'normalize_counts': False
        },
        'data_processing': {
            'batch_size': 1000,
            'max_features': 2898,
            'missing_value_strategy': 'zero_fill'
        },
        'model': {
            'random_forest': {
                'n_estimators': 200,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'class_weight': 'balanced_subsample',
                'random_state': 42
            }
        },
        'mongodb': {
            'connection_string': MONGODB_CONNECTION_STRING,
            'database_name': MONGODB_DB_NAME,
            'collection_name': MONGODB_COLLECTION_NAME,
            'features_collection': MONGODB_ML_FEATURES_COLLECTION,
            'anomaly_collection': MONGODB_ANOMALY_RESULTS_COLLECTION
        },
        'paths': {
            'model_dir': os.path.dirname(model_path_default),
            'log_dir': log_dir_path_default,
            'training_data_dir': 'Training_data',
            'qradar_rule_dir': 'Qradar_rule'
        }
    }

#### NEVER change the below!
# run the default to check the parameter
if __name__ == "__main__":
    print("Qradar_address_default:", Qradar_address_default)
    # Mask token output for safety
    masked_token = (
        ("****" + Qradar_token_default[-4:]) if Qradar_token_default and Qradar_token_default != "REPLACE_WITH_TOKEN" else "(unset)"
    )
    print("Qradar_token_default:", masked_token)
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
    print("SYSLOG_HEADER_BASE:", SYSLOG_HEADER_BASE)
    print("SYSLOG_HEADER_ML:", SYSLOG_HEADER_ML)
    print("SYSLOG_HEADER_LOG:", SYSLOG_HEADER_LOG)