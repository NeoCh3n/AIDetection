#!/usr/bin/env python3
"""
Export MongoDB data into Training_data-compatible CSV files.

This script supports two MongoDB source shapes:
- raw event documents (for example SIEM.NetworkConnection)
- aggregated detection windows (qradar_detection.qradar_sliding_windows)

In both cases it writes one CSV per 30-minute window using the existing
Training_data column format, so the current training pipeline can keep running
without downstream code changes.

Example usage:
    python mongodb/export_training_csv.py \
        --use-whitelist \
        --start-time "2025-07-29T00:00:00" \
        --end-time "2025-07-30T00:00:00" \
        --output-dir "./Training_data/normal"

    python mongodb/export_training_csv.py \
        --hostname "DESKTOP-64-EDR" \
        --hours-back 24 \
        --output-dir "./Training_data/normal/DESKTOP-64-EDR"

    python mongodb/export_training_csv.py \
        --db-name SIEM \
        --collection NetworkConnection \
        --source-schema raw \
        --start-time "2025-07-29T00:00:00" \
        --end-time "2025-07-30T00:00:00" \
        --output-dir "./Training_data/normal/DESKTOP-64-EDR" \
        --hostname "DESKTOP-64-EDR"

    python mongodb/export_training_csv.py \
        --use-whitelist \
        --start-time "2025-07-29T00:00:00" \
        --end-time "2025-07-30T00:00:00" \
        --output-dir "./Training_data/attack"

    python mongodb/export_training_csv.py \
        --hostname "DESKTOP-64-EDR" \
        --start-time "2025-07-29T00:00:00" \
        --end-time "2025-07-30T00:00:00" \
        --output-dir "./Training_data/attack/DESKTOP-64-EDR"

Python 3.6.8 Compatible
"""

import os
import sys
import json
import csv
import argparse
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pymongo import MongoClient


PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared_utils.time_utils import HKT, get_window_start_end, parse_qradar_timestamp


CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'mongodb_config.json')
CSV_COLUMNS = [
    'sysmon_hostname (custom)',
    'Custom Rule',
    'Log Source Time (Minimum)',
    'Count',
]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load MongoDB configuration from JSON."""
    with open(config_path, 'r') as handle:
        return json.load(handle)


def get_by_path(value: Any, path: str) -> Any:
    """Safely resolve a dotted path from nested dictionaries."""
    current = value
    if not path:
        return current

    for part in path.split('.'):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def iter_event_payloads(document: Dict[str, Any], payload_path: Optional[str]) -> Iterable[Dict[str, Any]]:
    """
    Yield one or more event payload dictionaries from a raw MongoDB document.

    If payload_path points to a dict, yield that dict.
    If payload_path points to a list, yield each dict entry in the list.
    Otherwise, yield the original document for top-level-field collections.
    """
    if payload_path:
        payload = get_by_path(document, payload_path)
        if isinstance(payload, dict):
            yield payload
            return
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    yield item
            return

    yield document


def normalize_count(raw_value: Any) -> Optional[int]:
    """Convert count-like values into integers."""
    if raw_value is None:
        return None

    text = str(raw_value).strip().replace(',', '')
    if not text:
        return None

    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def format_qradar_timestamp(dt: datetime) -> str:
    """Format datetime into the QRadar CSV timestamp string used by Training_data."""
    if dt.tzinfo is None:
        dt = HKT.localize(dt)
    else:
        dt = dt.astimezone(HKT)

    hour_12 = dt.hour % 12 or 12
    am_pm = dt.strftime('%p')
    return "{month} {day}, {year}, {hour}:{minute}:{second} {ampm}".format(
        month=dt.strftime('%b'),
        day=dt.day,
        year=dt.year,
        hour=hour_12,
        minute=dt.strftime('%M'),
        second=dt.strftime('%S'),
        ampm=am_pm,
    )


def parse_cli_datetime(value: str) -> datetime:
    """Parse CLI datetime in YYYY-MM-DDTHH:MM:SS form and localize to HKT."""
    dt = datetime.strptime(value, '%Y-%m-%dT%H:%M:%S')
    return HKT.localize(dt)


def build_query(hostname: Optional[str], hostname_query_field: Optional[str]) -> Dict[str, Any]:
    """Build the MongoDB query. Time filtering is applied after parsing event timestamps."""
    query = {}
    if hostname and hostname_query_field:
        query[hostname_query_field] = hostname
    return query


def build_detection_windows_query(hostname: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
    """Build the MongoDB query for detection_windows documents."""
    return {
        'host_triggers.' + hostname: {'$exists': True},
        'window_start': {
            '$gte': start_time,
            '$lt': end_time
        }
    }


def get_whitelist_hosts(config: Dict[str, Any]) -> List[str]:
    """Read configured whitelist hostnames from mongodb_config.json."""
    host_retention = config.get('host_retention', {})
    raw_hosts = host_retention.get('whitelist_hosts', [])
    if not isinstance(raw_hosts, list):
        return []

    hosts = []
    for host in raw_hosts:
        host_str = str(host).strip()
        if host_str:
            hosts.append(host_str)
    return hosts


def extract_training_row(
    payload: Dict[str, Any],
    hostname_field: str,
    rule_field: str,
    timestamp_field: str,
    count_field: str
) -> Optional[Tuple[str, str, datetime, int]]:
    """Extract one Training_data-compatible row from a payload."""
    hostname = get_by_path(payload, hostname_field)
    rule_id = get_by_path(payload, rule_field)
    timestamp_value = get_by_path(payload, timestamp_field)
    count_value = get_by_path(payload, count_field)

    if hostname is None or rule_id is None or timestamp_value is None:
        return None

    try:
        event_time = parse_qradar_timestamp(str(timestamp_value).strip())
    except Exception:
        return None

    count = normalize_count(count_value)
    if count is None:
        return None

    return str(hostname).strip(), str(rule_id).strip(), event_time, count


def normalize_window_time(raw_value: Any) -> Optional[datetime]:
    """Normalize a stored window timestamp into timezone-aware HKT datetime."""
    if raw_value is None:
        return None

    if isinstance(raw_value, datetime):
        if raw_value.tzinfo is None:
            return HKT.localize(raw_value)
        return raw_value.astimezone(HKT)

    text = str(raw_value).strip()
    if not text:
        return None

    parsed = None
    time_formats = [
        '%Y-%m-%dT%H:%M:%S.%f%z',
        '%Y-%m-%dT%H:%M:%S%z',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
    ]
    for time_format in time_formats:
        try:
            parsed = datetime.strptime(text, time_format)
            break
        except ValueError:
            continue
    if parsed is None:
        return None

    if parsed.tzinfo is None:
        return HKT.localize(parsed)
    return parsed.astimezone(HKT)


def detect_source_schema(collection_name: str, sample_doc: Optional[Dict[str, Any]]) -> str:
    """Infer the source schema from collection name and a sample document."""
    if collection_name == 'qradar_sliding_windows':
        return 'detection_windows'
    if sample_doc:
        if isinstance(sample_doc.get('host_triggers'), dict):
            return 'detection_windows'
        if 'events' in sample_doc:
            return 'raw'
    return 'raw'


def extract_window_rows(
    document: Dict[str, Any],
    hostname: str,
    start_time: datetime,
    end_time: datetime
) -> List[Tuple[datetime, Dict[str, Any]]]:
    """Expand one detection_windows document into Training_data-compatible rows."""
    host_triggers = document.get('host_triggers')
    if not isinstance(host_triggers, Mapping):
        return []

    host_payload = host_triggers.get(hostname)
    if not isinstance(host_payload, Mapping):
        return []

    window_start = normalize_window_time(document.get('window_start'))
    if window_start is None:
        return []

    rules = host_payload.get('rules')
    if not isinstance(rules, Mapping):
        return []

    rows = []
    for rule_id, count in rules.items():
        normalized = normalize_count(count)
        if normalized is None:
            continue

        rows.append((
            window_start,
            {
                'sysmon_hostname (custom)': hostname,
                'Custom Rule': str(rule_id),
                'Log Source Time (Minimum)': format_qradar_timestamp(window_start),
                'Count': normalized,
            }
        ))

    return rows


def ensure_dir(path: str) -> None:
    """Create output directory if it does not exist."""
    if not os.path.isdir(path):
        os.makedirs(path)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Export MongoDB data into Training_data-compatible 30-minute CSV files'
    )
    parser.add_argument('--config', default=CONFIG_PATH,
                        help='Path to mongodb_config.json')
    parser.add_argument('--db-name', default=None,
                        help='MongoDB database name for the source data')
    parser.add_argument('--collection', default=None,
                        help='MongoDB collection name for the source data')
    parser.add_argument('--hostname', default=None,
                        help='Hostname to export exactly as stored in MongoDB')
    parser.add_argument('--use-whitelist', action='store_true',
                        help='Export all hostnames from host_retention.whitelist_hosts in config')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to write one CSV per 30-minute window')
    parser.add_argument('--hours-back', type=int, default=None,
                        help='Look back N hours from now')
    parser.add_argument('--start-time', default=None,
                        help='Range start in YYYY-MM-DDTHH:MM:SS (HKT)')
    parser.add_argument('--end-time', default=None,
                        help='Range end in YYYY-MM-DDTHH:MM:SS (HKT)')
    parser.add_argument('--window-size-minutes', type=int, default=30,
                        help='Window size in minutes (default: 30)')
    parser.add_argument('--source-schema', choices=['auto', 'raw', 'detection_windows'], default='auto',
                        help='Mongo source schema to export from (default: auto)')
    parser.add_argument('--hostname-query-field', default='events.sysmon_hostname (custom)',
                        help='Field path used in MongoDB query for hostname filtering')
    parser.add_argument('--payload-path', default='events',
                        help='Field path containing the raw event payload; use empty string for top-level fields')
    parser.add_argument('--hostname-field', default='sysmon_hostname (custom)',
                        help='Field name inside the payload for hostname')
    parser.add_argument('--rule-field', default='Custom Rule',
                        help='Field name inside the payload for rule ID')
    parser.add_argument('--timestamp-field', default='Log Source Time (Minimum)',
                        help='Field name inside the payload for QRadar timestamp')
    parser.add_argument('--count-field', default='Count',
                        help='Field name inside the payload for event count')
    parser.add_argument('--limit', type=int, default=0,
                        help='Optional Mongo cursor limit; 0 means no limit')
    args = parser.parse_args()

    if args.hours_back is None and (not args.start_time or not args.end_time):
        parser.error('Provide either --hours-back or both --start-time and --end-time')

    if args.hours_back is not None and (args.start_time or args.end_time):
        parser.error('Use either --hours-back or --start-time/--end-time, not both')

    if args.use_whitelist and args.hostname:
        parser.error('Do not pass --hostname together with --use-whitelist')
    if not args.use_whitelist and not args.hostname:
        parser.error('Provide --hostname, or use --use-whitelist')

    if args.hours_back is not None:
        end_time = datetime.now(HKT)
        start_time = end_time - timedelta(hours=args.hours_back)
    else:
        start_time = parse_cli_datetime(args.start_time)
        end_time = parse_cli_datetime(args.end_time)

    if end_time <= start_time:
        parser.error('--end-time must be later than --start-time')

    config = load_config(args.config)
    connection_string = config['mongodb']['connection_string']
    detection_db_name = config['mongodb'].get('db_name', 'qradar_detection')
    detection_collection = config.get('collections', {}).get('detection_windows', 'qradar_sliding_windows')
    db_name = args.db_name or detection_db_name
    collection_name = args.collection or detection_collection
    hostnames = [args.hostname]
    if args.use_whitelist:
        hostnames = get_whitelist_hosts(config)
        if not hostnames:
            parser.error('No hostnames found in host_retention.whitelist_hosts')

    print('Exporting MongoDB data to Training_data-compatible CSV files...')
    print('  DB: {}'.format(db_name))
    print('  Collection: {}'.format(collection_name))
    print('  Hostnames: {}'.format(', '.join(hostnames)))
    print('  Time range: {} -> {}'.format(start_time.isoformat(), end_time.isoformat()))

    client = MongoClient(connection_string)
    db = client[db_name]
    collection = db[collection_name]

    sample_doc = collection.find_one()
    source_schema = args.source_schema
    if source_schema == 'auto':
        source_schema = detect_source_schema(collection_name, sample_doc)
    print('  Source schema: {}'.format(source_schema))

    total_raw_docs = 0
    total_matched_rows = 0
    total_files_written = 0

    try:
        for hostname in hostnames:
            if source_schema == 'detection_windows':
                query = build_detection_windows_query(hostname, start_time, end_time)
            else:
                query = build_query(hostname, args.hostname_query_field)
            cursor = collection.find(query)
            if args.limit and args.limit > 0:
                cursor = cursor.limit(args.limit)

            window_rows = defaultdict(list)
            raw_docs = 0
            matched_rows = 0

            for document in cursor:
                raw_docs += 1
                if source_schema == 'detection_windows':
                    extracted_rows = extract_window_rows(
                        document=document,
                        hostname=hostname,
                        start_time=start_time,
                        end_time=end_time
                    )
                    for event_time, row in extracted_rows:
                        bucket_start, _ = get_window_start_end(event_time, args.window_size_minutes)
                        window_rows[bucket_start].append(row)
                        matched_rows += 1
                else:
                    for payload in iter_event_payloads(document, args.payload_path or None):
                        extracted = extract_training_row(
                            payload=payload,
                            hostname_field=args.hostname_field,
                            rule_field=args.rule_field,
                            timestamp_field=args.timestamp_field,
                            count_field=args.count_field
                        )
                        if extracted is None:
                            continue

                        payload_hostname, rule_id, event_time, count = extracted
                        if payload_hostname != hostname:
                            continue
                        if event_time < start_time or event_time >= end_time:
                            continue

                        window_start, _ = get_window_start_end(event_time, args.window_size_minutes)
                        window_rows[window_start].append({
                            'sysmon_hostname (custom)': payload_hostname,
                            'Custom Rule': rule_id,
                            'Log Source Time (Minimum)': format_qradar_timestamp(event_time),
                            'Count': count,
                        })
                        matched_rows += 1

            host_output_dir = args.output_dir
            if args.use_whitelist:
                host_output_dir = os.path.join(args.output_dir, hostname)

            ensure_dir(host_output_dir)

            files_written = 0
            for window_start in sorted(window_rows.keys()):
                rows = window_rows[window_start]
                if not rows:
                    continue

                filename = '{hostname}_{window}.csv'.format(
                    hostname=hostname,
                    window=window_start.strftime('%Y-%m-%d_%H-%M-%S')
                )
                output_path = os.path.join(host_output_dir, filename)

                with open(output_path, 'w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)

                files_written += 1

            total_raw_docs += raw_docs
            total_matched_rows += matched_rows
            total_files_written += files_written

            print('  Host {}: scanned {} docs, exported {} rows into {} files'.format(
                hostname, raw_docs, matched_rows, files_written
            ))
    finally:
        client.close()

    print('  Raw documents scanned: {}'.format(total_raw_docs))
    print('  Exported rows: {}'.format(total_matched_rows))
    print('  CSV files written: {}'.format(total_files_written))
    print('  Output directory: {}'.format(os.path.abspath(args.output_dir)))

    return 0


if __name__ == '__main__':
    exit(main())
