import time
import datetime
import os
import json
import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from . import config

# Use centralized configuration
config_dict = config.get_config()
fetch_data_frequency = config.fetch_data_frequency_default
log_dir_path = config.log_dir_path_default
log_destination_address = config.log_destination_address_default
log_destination_port = config.log_destination_port_default
data_retention_days = config.DATA_RETENTION_DAYS
header_ml = config.SYSLOG_HEADER_ML
header_log = config.SYSLOG_HEADER_LOG

SUPPRESS_STDOUT = True  # avoid duplicate console noise; rely on app logger

def run_log(level, message, payload = None):
    # Ensure log directory exists
    Path(log_dir_path).mkdir(parents=True, exist_ok=True)

    date_lastweek = str(datetime.datetime.now() - datetime.timedelta(days=data_retention_days))[:10]
    date_15mins_ago = str(datetime.datetime.now() - datetime.timedelta(minutes=15))[:10]
    time_now = datetime.datetime.today().strftime('%Y-%b-%d %H:%M:%S')

    filepath_today = log_dir_path + str(date_15mins_ago) + '.log'
    filepath_lastweek = log_dir_path + str(date_lastweek) + '.log'

    # write logs
    total_minute = int(datetime.datetime.now().hour) * 60 + int(datetime.datetime.now().minute)
    session_count = total_minute // fetch_data_frequency
    total_session = int(1440 / fetch_data_frequency)

    with open(filepath_today, "a+") as logfile_today:
        # delete the logs of last week
        if os.path.exists(filepath_lastweek):
            os.remove(filepath_lastweek)
            logfile_today.write("| " + level.ljust(10) + " | " + str(time_now) + " | " + "Session: " + str(session_count) + "/" + str(total_session) + " | 0. " + filepath_lastweek + " deleted\n")
        # delete any logs older than retention window
        cutoff = datetime.datetime.now().date() - datetime.timedelta(days=data_retention_days)
        for f in Path(log_dir_path).glob("*.log"):
            try:
                date_str = f.stem  # expecting YYYY-MM-DD
                file_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                if file_date < cutoff and f.exists():
                    f.unlink()
                    logfile_today.write("| " + level.ljust(10) + " | " + str(time_now) + " | " + "Session: " + str(session_count) + "/" + str(total_session) + " | old log deleted: " + str(f) + "\n")
            except Exception:
                # ignore files not following the naming convention
                continue
        log_message = "| " + level.ljust(10) + " | " + str(time_now) + " | " + "Session: " + str(session_count) + "/" + str(total_session) + " | " + message + "\n"
        if payload is not None:
            # Keep payload on a single line to avoid breaking syslog formatting
            payload_str = str(payload).replace("\n", " ")
            log_message += " | Payload: " + payload_str + "\n"
        logfile_today.write(log_message)
        # Optional stdout printing (disabled by default to avoid duplicates)
        if not SUPPRESS_STDOUT:
            print(log_message)

    # send the logs to Qradar
    # Sanitize syslog message to a single line and cap size
    def _sanitize(msg: str, max_len: int = 8192) -> str:
        msg1 = msg.replace("\n", " ").replace("\r", " ")
        return msg1[:max_len]

    try:
        if level == header_ml:
            _send_syslog(log_destination_address, log_destination_port, header_ml, _sanitize(message))
        else:
            _send_syslog(log_destination_address, log_destination_port, header_log, _sanitize(log_message))
    except Exception as e:
        # Fallback: write a local error note
        with open(filepath_today, "a+") as logfile_today:
            logfile_today.write(f"| {'ERROR'.ljust(10)} | {time_now} | Syslog send failed: {e}\n")

def log_detection(hostname, window_id, prediction, confidence, timestamp=None):
    """
    Log attack detection events with standardized format.
    
    Args:
        hostname (str): Hostname where detection occurred
        window_id (str): Time window identifier
        prediction (str): Prediction result ('malicious' or 'normal')
        confidence (float): Confidence score (0-1)
        timestamp (datetime, optional): Timestamp of detection
    """
    if timestamp is None:
        timestamp = datetime.datetime.now()
    
    message = f"ATTACK_DETECTION | hostname={hostname} | window_id={window_id} | prediction={prediction} | confidence={confidence:.4f}"
    run_log("DETECTION", message)

def log_shap_results(hostname, window_id, top_rules, shap_values, prediction, confidence):
    """
    Log SHAP explanation results for attack detections.
    
    Args:
        hostname (str): Hostname where detection occurred
        window_id (str): Time window identifier
        top_rules (list): List of top contributing rule IDs
        shap_values (list): Corresponding SHAP values
        prediction (str): Prediction result ('malicious' or 'normal')
        confidence (float): Confidence score (0-1)
    """
    shap_data = {
        "hostname": hostname,
        "window_id": window_id,
        "prediction": prediction,
        "confidence": confidence,
        "top_rules": top_rules,
        "shap_values": shap_values,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    message = f"SHAP_EXPLANATION | hostname={hostname} | window_id={window_id}"
    run_log("EXPLAIN", message, payload=shap_data)

def log_pipeline_event(event_type, message, metadata=None):
    """
    Log pipeline events for training and detection modes.
    
    Args:
        event_type (str): Type of pipeline event
        message (str): Event description
        metadata (dict, optional): Additional metadata
    """
    if metadata is None:
        metadata = {}
    metadata['event_type'] = event_type
    run_log("PIPELINE", message, payload=metadata)

def log_feature_importance(rule_importance_dict, top_n=20):
    """
    Log top contributing rules for attack detection.
    
    Args:
        rule_importance_dict (dict): Dictionary mapping rule IDs to importance scores
        top_n (int): Number of top rules to log
    """
    sorted_rules = sorted(rule_importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    importance_data = {
        "top_rules": sorted_rules,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    message = f"FEATURE_IMPORTANCE | top_{top_n}_rules"
    run_log("IMPORTANCE", message, payload=importance_data)

# SysLogHandler cache and functions
_LOGGER_NAME = "air.syslog"
_handler_cache = {}

def _get_logger(address: str, port: int, header: str) -> logging.Logger:
    """Create or reuse a dedicated logger with a SysLogHandler.
    We avoid touching the root logger to not interfere with application logging.
    """
    key = (address, port, header)
    logger = logging.getLogger(f"{_LOGGER_NAME}.{address}:{port}.{header}")
    logger.setLevel(logging.INFO)

    if key not in _handler_cache:
        handler = logging.handlers.SysLogHandler(address=(address, port))
        # Use dynamic timestamp per message
        formatter = logging.Formatter(f'%(asctime)s {header} %(message)s', datefmt='%b %d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        _handler_cache[key] = handler
    else:
        # Ensure the handler is attached (in case logger was recreated)
        handler = _handler_cache[key]
        if handler not in logger.handlers:
            logger.addHandler(handler)
    # Prevent propagation to root logger to avoid duplicate StreamHandler output
    logger.propagate = False

    return logger

def _send_syslog(address: str = config.log_destination_address_default,
                 port: int = config.log_destination_port_default,
                 header: str = config.SYSLOG_HEADER_LOG,
                 message: Optional[str] = None) -> None:
    """Send a syslog message to the syslog server."""
    if message is None:
        return
    # Ensure single-line message
    msg = str(message).replace('\n', ' ').replace('\r', ' ')
    logger = _get_logger(address, port, header)
    try:
        logger.info(msg)
    except Exception:
        # Best-effort: drop errors silently to avoid crashing callers
        pass

if __name__ == "__main__":
    # Test basic logging
    run_log("DEBUG", "debug log")
    time.sleep(1)
    run_log("INFO", "info log")
    time.sleep(1)
    run_log("WARNING", "warning log")
    time.sleep(1)
    run_log("ERROR", "error log")
    time.sleep(1)
    run_log("CRITICAL", "critical log")
    
    # Test attack detection logging
    log_detection("test-host", "2024-01-01-1200", "malicious", 0.95)
    
    # Test SHAP explanation logging
    log_shap_results("test-host", "2024-01-01-1200", [1001, 1002, 1003], [0.45, 0.32, 0.23], "malicious", 0.95)
    
    # Test pipeline logging
    log_pipeline_event("TRAINING_START", "Started training pipeline", {"mode": "train"})
    
    # Test feature importance logging
    sample_importance = {f"rule_{i}": 0.01 * i for i in range(1, 21)}
    log_feature_importance(sample_importance)
