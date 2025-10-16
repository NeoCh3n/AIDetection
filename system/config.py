import json
import os
from typing import Any, Dict

"""
Centralized configuration for the unified pipeline.

This module bridges legacy callers (system.logging_utils, tests, some pipeline
modules) with the modern JSON config at pipeline/config.json. It exposes a small set of constants for logging/syslog and a
get_config() function returning a consolidated dictionary used by components
like feature_generator and tests.

Backwards compatibility helpers are included to tolerate older access patterns
such as config.get('key', default).
"""

# -------------------------
# Syslog / logging defaults
# -------------------------
DATA_RETENTION_DAYS = int(os.getenv("DATA_RETENTION_DAYS", "7"))

# Frequency for periodic scheduling (minutes). Used by some logs/tests.
fetch_data_frequency_default = int(os.getenv("FETCH_FREQUENCY_MIN", "30"))

# Where local logs are written
log_dir_path_default = os.getenv("LOG_DIR", os.path.join("running_log/"))

# Syslog target and headers
log_destination_address_default = os.getenv("SYSLOG_ADDRESS", "192.168.153.123")
log_destination_port_default = int(os.getenv("SYSLOG_PORT", "514"))

SYSLOG_HEADER_BASE = os.getenv("SYSLOG_HEADER_BASE", "AIR")
SYSLOG_HEADER_ML = os.getenv("SYSLOG_HEADER_ML", f"{SYSLOG_HEADER_BASE}-RF")
SYSLOG_HEADER_LOG = os.getenv("SYSLOG_HEADER_LOG", f"{SYSLOG_HEADER_BASE}-RF")


# -------------------------
# QRadar defaults (env-based)
# -------------------------
Qradar_address_default = os.getenv("QRADAR_ADDRESS", "192.168.153.123")
Qradar_token_default = os.getenv("QRADAR_API_TOKEN", "REPLACE_WITH_TOKEN")
request_header_default = {
    'SEC': Qradar_token_default,
    'Version': os.getenv('QRADAR_API_VERSION', '20.0'),
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Connection': 'Close',
}


# -------------------------
# Pipeline JSON config loader
# -------------------------
def _load_json_config() -> Dict[str, Any]:
    """Load pipeline/config.json if present, else return {}."""
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pipeline', 'config.json')
    try:
        with open(cfg_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def get_config() -> Dict[str, Any]:
    """
    Return consolidated configuration reflecting AGENTS.md.

    - Prefers pipeline/config.json values when present.
    - Provides sensible defaults for missing sections.
    - Includes paths.model_dir for tests and modules expecting it.
    """
    file_cfg = _load_json_config()

    # Defaults aligned with AGENTS.md
    defaults: Dict[str, Any] = {
        'rule_manager': {
            'mode': 'file',
            'rule_file_path': os.path.join('Qradar_rule', 'production_rules.txt'),
            'uat_rule_file_path': os.path.join('Qradar_rule', 'uat_rules.txt'),
            'mapping_file_path': os.path.join('Qradar_rule', 'uat_to_prod_mapping.json'),
            'environment': os.getenv('ENVIRONMENT', 'prod')
        },
        'feature_engineering': {
            'expected_rules': 1128,
            'window_size_minutes': 30,
            'log_transform': True,
            'normalize_counts': False,
        },
        'model': {
            'random_forest': {
                'n_estimators': 200,
                'class_weight': 'balanced_subsample',
                'max_features': 'sqrt',
                'random_state': 42,
                'n_jobs': -1,
            }
        },
        'mongodb': {
            'connection_string': 'mongodb://localhost:27017/',
            'database_name': 'qradar_ml',
            'collection_name': 'detection_data',
        },
        'paths': {
            'model_dir': 'model',
            'log_dir': 'running_log',
            'training_data_dir': 'Training_data',
            'qradar_rule_dir': 'Qradar_rule',
        },
    }

    # Merge relevant pieces from pipeline/config.json when present
    merged = defaults.copy()

    # training.model_path informs paths.model_dir if available
    training_cfg = file_cfg.get('training', {}) if isinstance(file_cfg, dict) else {}
    if isinstance(training_cfg, dict):
        model_path = training_cfg.get('model_path')
        if isinstance(model_path, str):
            merged['paths']['model_dir'] = os.path.dirname(model_path) or 'model'

    # detection.mongodb_config maps into mongodb defaults
    detect_cfg = file_cfg.get('detection', {}) if isinstance(file_cfg, dict) else {}
    if isinstance(detect_cfg, dict):
        mdb = detect_cfg.get('mongodb_config', {})
        if isinstance(mdb, dict):
            merged['mongodb']['connection_string'] = mdb.get('connection_string', merged['mongodb']['connection_string'])
            merged['mongodb']['database_name'] = mdb.get('database', merged['mongodb']['database_name'])
            merged['mongodb']['collection_name'] = mdb.get('collection', merged['mongodb']['collection_name'])

    # paths override
    paths_cfg = file_cfg.get('paths', {}) if isinstance(file_cfg, dict) else {}
    if isinstance(paths_cfg, dict):
        merged['paths']['model_dir'] = paths_cfg.get('model_output', merged['paths']['model_dir']).replace('./', '') if paths_cfg.get('model_output') else merged['paths']['model_dir']
        merged['paths']['log_dir'] = paths_cfg.get('logs', merged['paths']['log_dir']).replace('./', '') if paths_cfg.get('logs') else merged['paths']['log_dir']
        merged['paths']['training_data_dir'] = paths_cfg.get('training_data', merged['paths']['training_data_dir']).replace('./', '') if paths_cfg.get('training_data') else merged['paths']['training_data_dir']

    return merged


def get(key: str, default: Any = None) -> Any:
    """
    Backwards-compatible accessor for callers that mistakenly treat this
    module as a mapping. Returns a top-level key from get_config(), or default.
    """
    try:
        return get_config().get(key, default)
    except Exception:
        return default


if __name__ == "__main__":
    cfg = get_config()
    print("Model dir:", cfg['paths']['model_dir'])
    print("Log dir:", cfg['paths']['log_dir'])
    print("MongoDB:", cfg['mongodb'])
    print("RF params:", cfg['model']['random_forest'])
