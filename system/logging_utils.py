import time
import datetime
import os
import json
import logging
import logging.handlers
import textwrap
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, Tuple
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

# Cache for root-level file handler to avoid duplicates
_ROOT_FILE_HANDLER_PATH: Optional[str] = None
_ROOT_FILE_HANDLER: Optional[logging.Handler] = None


def _sanitize_message(msg: str, max_len: int = 8192, allow_multiline: bool = False) -> str:
    """
    Sanitize a message before transmitting to syslog.

    Args:
        msg: The message to sanitize.
        max_len: Maximum payload size allowed.
        allow_multiline: If True, preserve newline characters.
    """
    sanitized = str(msg).replace("\r", " ")
    if not allow_multiline:
        sanitized = sanitized.replace("\n", " ")
    return sanitized[:max_len]


def _format_top_feature_entry(entry: Any) -> Dict[str, Any]:
    """
    Normalize a feature entry into a consistent structure for rendering.
    """
    feature_id: Optional[str] = None
    label: Optional[str] = None
    importance: Optional[float] = None

    if isinstance(entry, dict):
        feature_id = entry.get('feature') or entry.get('feature_id') or entry.get('rule_id') or entry.get('name')
        label = entry.get('label') or entry.get('rule_name') or entry.get('description')
        importance = entry.get('importance')
        if importance is None:
            importance = entry.get('score') or entry.get('value')
    elif isinstance(entry, (list, tuple)):
        tpl: Tuple[Any, ...] = tuple(entry)
        if tpl:
            feature_id = tpl[0]
        if len(tpl) > 1:
            importance = tpl[1]
        if len(tpl) > 2:
            label = tpl[2]
    else:
        feature_id = str(entry)

    display_parts = []
    if feature_id is not None:
        display_parts.append(str(feature_id))
    if label:
        display_parts.append(str(label))

    display = " | ".join(display_parts) if display_parts else str(entry)
    display = textwrap.shorten(display, width=80, placeholder="...")

    return {
        'feature_id': feature_id,
        'label': label,
        'display': display,
        'importance': importance,
    }


def _format_detection_payload(
    hostname: str,
    window_id: str,
    prediction: str,
    confidence: Optional[float],
    timestamp: datetime.datetime,
    model_name: Optional[str] = None,
    feature_count: Optional[int] = None,
    instances_analyzed: Optional[int] = None,
    top_features: Optional[Sequence[Any]] = None,
) -> str:
    """
    Construct a LEEF-formatted detection payload for QRadar.
    """
    model_display = model_name or "RandomForestClassifier"
    instance_display = instances_analyzed if instances_analyzed is not None else 1
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    confidence_pct: Optional[float] = None
    if confidence is not None:
        try:
            conf_val = float(confidence)
            confidence_pct = conf_val * 100 if conf_val <= 1.5 else conf_val
        except (TypeError, ValueError):
            confidence_pct = None

    if isinstance(prediction, str):
        prediction_label = prediction.strip().capitalize()
    else:
        prediction_label = str(prediction)

    confidence_str = f"{confidence_pct:.1f}%" if confidence_pct is not None else "N/A"

    processed_features = []
    if top_features:
        processed_features = [_format_top_feature_entry(entry) for entry in top_features]

    def _clean_value(value: Any) -> str:
        text = str(value)
        return text.replace("\t", " ").replace("\n", " ").replace("=", ":")

    top_feature_summary = ""
    if processed_features:
        tokens = []
        for idx, feature in enumerate(processed_features, start=1):
            importance = feature.get('importance')
            if importance is not None:
                try:
                    importance_val = float(importance)
                    imp_label = f"{importance_val:.4f}"
                except (TypeError, ValueError):
                    imp_label = str(importance)
            else:
                imp_label = "N/A"
            tokens.append(f"{idx}:{feature['display']}:{imp_label}")
        top_feature_summary = ";".join(tokens)

    most_critical = ""
    if processed_features:
        primary = processed_features[0]
        importance_primary = primary.get('importance')
        if importance_primary is not None:
            try:
                primary_score = f"{float(importance_primary):.4f}"
            except (TypeError, ValueError):
                primary_score = str(importance_primary)
        else:
            primary_score = "N/A"
        most_critical = f"{primary.get('display')}:{primary_score}"

    feature_count_str = str(feature_count) if feature_count is not None else "unknown"

    extensions = [
        ("name", "Malicious Threat Detection"),
        ("hostname", hostname),
        ("windowId", window_id),
        ("model", model_display),
        ("instances", instance_display),
        ("features", feature_count_str),
        ("timestamp", timestamp_str),
        ("prediction", prediction_label),
        ("confidencePct", confidence_str),
        ("topFeatures", top_feature_summary or "n/a"),
        ("mostCritical", most_critical or "n/a"),
    ]

    extension_str = "\t".join(f"{key}={_clean_value(value)}" for key, value in extensions)

    header = "LEEF:2.0|AI4|RFDetector|1.0|ATTACK_DETECTION|High|"

    return header + extension_str


def _normalize_top_rules_payload(payload: Any) -> Any:
    """
    Ensure the top_rules_by_count payload entries include indexed keys
    (e.g., rule_id-1, value-1) to satisfy QRadar formatting expectations.
    """
    if not isinstance(payload, dict):
        return payload

    top_rules_raw = payload.get('top_rules_by_count')
    if not isinstance(top_rules_raw, (list, tuple)) or not top_rules_raw:
        return payload

    transformed_rules = []
    for idx, entry in enumerate(top_rules_raw, start=1):
        if isinstance(entry, dict):
            normalized_entry = {
                f'rule_id-{idx}': entry.get('rule_id'),
                f'value-{idx}': entry.get('value'),
            }
            rule_name_val = entry.get('rule_name')
            if rule_name_val is not None:
                normalized_entry[f'rule_name-{idx}'] = rule_name_val
        else:
            normalized_entry = {f'value-{idx}': entry}
        transformed_rules.append(normalized_entry)

    payload_copy = dict(payload)
    payload_copy['top_rules_by_count'] = transformed_rules
    return payload_copy


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
    session_str = f"Session: {session_count}/{total_session}"
    level_label = level.ljust(10)
    level_pad = " " * len(level_label)
    time_pad = " " * len(time_now)
    session_pad = " " * len(session_str)
    base_prefix = f"| {level_label} | {time_now} | {session_str} | "
    continuation_prefix = f"| {level_pad} | {time_pad} | {session_pad} | "

    message_str = str(message)
    message_lines = message_str.splitlines() if message_str else [""]

    with open(filepath_today, "a+") as logfile_today:
        # delete the logs of last week
        if os.path.exists(filepath_lastweek):
            os.remove(filepath_lastweek)
            logfile_today.write(f"{base_prefix}0. {filepath_lastweek} deleted\n")
        # delete any logs older than retention window
        cutoff = datetime.datetime.now().date() - datetime.timedelta(days=data_retention_days)
        for f in Path(log_dir_path).glob("*.log"):
            try:
                date_str = f.stem  # expecting YYYY-MM-DD
                file_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                if file_date < cutoff and f.exists():
                    f.unlink()
                    logfile_today.write(f"{base_prefix}old log deleted: {str(f)}\n")
            except Exception:
                # ignore files not following the naming convention
                continue

        log_lines = [base_prefix + (message_lines[0] if message_lines else "")]
        for extra_line in message_lines[1:]:
            log_lines.append(continuation_prefix + extra_line)

        payload_to_log = payload
        if payload is not None:
            try:
                payload_to_log = _normalize_top_rules_payload(payload)
            except Exception:
                payload_to_log = payload
            try:
                payload_str = json.dumps(payload_to_log, default=str)
            except Exception:
                payload_str = str(payload_to_log).replace("\n", " ")
            log_lines.append(continuation_prefix + "Payload: " + payload_str)

        log_message = "\n".join(log_lines) + "\n"
        logfile_today.write(log_message)
        # Optional stdout printing (disabled by default to avoid duplicates)
        if not SUPPRESS_STDOUT:
            print(log_message)

    # send the logs to Qradar
    try:
        is_ml_header = level == header_ml
        syslog_payload = message_str if is_ml_header else log_message
        sanitized = _sanitize_message(syslog_payload, allow_multiline=is_ml_header)
        header_to_use = header_ml if is_ml_header else header_log
        _send_syslog(log_destination_address, log_destination_port, header_to_use, sanitized)
    except Exception as e:
        # Fallback: write a local error note
        with open(filepath_today, "a+") as logfile_today:
            logfile_today.write(f"| {'ERROR'.ljust(10)} | {time_now} | Syslog send failed: {e}\n")


def _daily_log_filepath() -> str:
    """Compute the daily log file path consistent with run_log naming."""
    Path(log_dir_path).mkdir(parents=True, exist_ok=True)
    date_15mins_ago = str(datetime.datetime.now() - datetime.timedelta(minutes=15))[:10]
    # Preserve behavior where log_dir_path may include a trailing slash
    base_dir = log_dir_path.rstrip("/\\")
    return os.path.join(base_dir, f"{date_15mins_ago}.log")


def setup_global_daily_file_logging(level: int = logging.INFO, include_stdout: bool = True) -> None:
    """
    Attach a FileHandler to the root logger that writes all Python logging
    records to running_log/YYYY-MM-DD.log. This ensures that logs emitted via
    the standard logging module and via this utility converge to the same file.

    Args:
        level: Logging level to apply to the root logger.
        include_stdout: If True, ensure a StreamHandler exists for console output.
    """
    global _ROOT_FILE_HANDLER_PATH, _ROOT_FILE_HANDLER

    target_path = _daily_log_filepath()

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Ensure a single FileHandler for the daily log
    handler_exists = False
    for h in list(root_logger.handlers):
        if isinstance(h, logging.FileHandler):
            try:
                if os.path.abspath(getattr(h, 'baseFilename', '')) == os.path.abspath(target_path):
                    handler_exists = True
                    _ROOT_FILE_HANDLER_PATH = target_path
                    _ROOT_FILE_HANDLER = h
                    break
            except Exception:
                continue

    if not handler_exists:
        try:
            fh = logging.FileHandler(target_path, mode='a', encoding='utf-8', delay=True)
            fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(fmt)
            root_logger.addHandler(fh)
            _ROOT_FILE_HANDLER_PATH = target_path
            _ROOT_FILE_HANDLER = fh
        except Exception:
            # Best effort; do not raise to callers
            pass

    # Ensure console output if requested
    if include_stdout:
        has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                         for h in root_logger.handlers)
        if not has_stream:
            try:
                sh = logging.StreamHandler()
                sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                root_logger.addHandler(sh)
            except Exception:
                pass

def log_detection(
    hostname: str,
    window_id: str,
    prediction: str,
    confidence: Optional[float] = None,
    timestamp: Optional[datetime.datetime] = None,
    model_name: Optional[str] = None,
    feature_count: Optional[int] = None,
    instances_analyzed: Optional[int] = None,
    top_features: Optional[Sequence[Any]] = None,
):
    """
    Log attack detection events with standardized format.
    
    Args:
        hostname (str): Hostname where detection occurred
        window_id (str): Time window identifier
        prediction (str): Prediction result ('malicious' or 'normal')
        confidence (float, optional): Confidence score (0-1). Defaults to None.
        timestamp (datetime, optional): Timestamp of detection
        model_name (str, optional): Name of the model producing the prediction
        feature_count (int, optional): Number of features considered
        instances_analyzed (int, optional): Number of instances evaluated
        top_features (Sequence, optional): Iterable of top contributing features with metadata
    """
    if timestamp is None:
        timestamp = datetime.datetime.now()

    message = _format_detection_payload(
        hostname=hostname,
        window_id=window_id,
        prediction=prediction,
        confidence=confidence,
        timestamp=timestamp,
        model_name=model_name,
        feature_count=feature_count,
        instances_analyzed=instances_analyzed,
        top_features=top_features,
    )
    run_log(header_ml, message)

def log_shap_results(hostname, source_ip, window_id, top_rules, shap_values, prediction, confidence):
    """
    Log SHAP explanation results for attack detections.
    
    Args:
        hostname (str): Hostname where detection occurred
        source_ip (str): Source IP address
        window_id (str): Time window identifier
        top_rules (list): List of top contributing rule IDs
        shap_values (list): Corresponding SHAP values
        prediction (str): Prediction result ('malicious' or 'normal')
        confidence (float): Confidence score (0-1)
    """
    shap_data = {
        "hostname": hostname,
        "source_ip": source_ip,
        "window_id": window_id,
        "prediction": prediction,
        "confidence": confidence,
        "top_rules": top_rules,
        "shap_values": shap_values,
        "timestamp": datetime.datetime.now().isoformat()
    }

    message = f"SHAP_EXPLANATION | hostname={hostname} source_ip={source_ip} | window_id={window_id}"
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
    log_shap_results("test-host", "0.0.0.0", "2024-01-01-1200", [1001, 1002, 1003], [0.45, 0.32, 0.23], "malicious", 0.95)
    
    # Test pipeline logging
    log_pipeline_event("TRAINING_START", "Started training pipeline", {"mode": "train"})
    
    # Test feature importance logging
    sample_importance = {f"rule_{i}": 0.01 * i for i in range(1, 21)}
    log_feature_importance(sample_importance)
