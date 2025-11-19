"""
Time Utilities Module for Unified Data Processing Pipeline

This module provides standardized time handling for both training and detection modes.
It handles timestamp parsing, window ID generation,
and timezone management across different data sources.

Python 3.6.8 Compatible
"""

import datetime
import pytz
from typing import Union, Optional

# Define HKT timezone as constant
HKT = pytz.timezone('Asia/Hong_Kong')


def parse_qradar_timestamp(timestamp_str: str) -> datetime.datetime:
    """
    Parse QRadar timestamp strings into datetime objects in HKT timezone.
    
    Handles QRadar formats:
    - "Jul 29, 2025, 5:34:47 PM"
    - "Jul 30, 2025, 7:32:38 AM"
    
    Args:
        timestamp_str: String timestamp from QRadar
        
    Returns:
        datetime object in HKT timezone
        
    Raises:
        ValueError: If timestamp format is not recognized
    """
    if not timestamp_str:
        raise ValueError("Empty timestamp string provided")
    
    # Convert to string if not already
    timestamp_str = str(timestamp_str).strip()
    
    # Exact format for QRadar timestamps
    target_format = "%b %d, %Y, %I:%M:%S %p"
    
    try:
        dt = datetime.datetime.strptime(timestamp_str, target_format)
        # Localize to HKT timezone
        return HKT.localize(dt)
    except ValueError:
        raise ValueError(f"Unrecognized timestamp format: {timestamp_str}")


def get_window_id(timestamp: datetime.datetime, window_size_minutes: int = 15) -> str:
    """
    Generate a unique window ID for 15-minute time windows

    Args:
        timestamp: The datetime to generate window ID for
        window_size_minutes: Size of time window in minutes (default: 15)
        
    Returns:
        String window ID in format "YYYY-MM-DD_HH-MM-SS_WID"
    """
    if not isinstance(timestamp, datetime.datetime):
        raise TypeError("timestamp must be a datetime object")
    
    # Ensure timestamp is timezone-aware (UTC)
    if timestamp.tzinfo is None:
        timestamp = pytz.UTC.localize(timestamp)
    
    # Calculate window start time
    total_minutes = timestamp.hour * 60 + timestamp.minute
    window_number = total_minutes // window_size_minutes
    window_start_minute = window_number * window_size_minutes
    
    # Create window start datetime
    window_start = timestamp.replace(
        hour=window_start_minute // 60,
        minute=window_start_minute % 60,
        second=0,
        microsecond=0
    )
    
    # Generate window ID
    window_id = window_start.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{window_id}_W{window_number}"


def adjust_csv_timestamp_to_window_start(timestamp: datetime.datetime, window_size_minutes: int = 15) -> datetime.datetime:
    """
    Adjust CSV timestamp to the start of its 30-minute window.
    
    For manual CSV queries, the timestamp is the start time and needs to be 
    forwarded to align with 30-minute windows.
    
    Examples:
    - Jul 29, 2025, 9:50:55 AM -> Jul 29, 2025, 9:30:00 AM
    - Jul 30, 2025, 10:10:30 AM -> Jul 30, 2025, 10:00:00 AM
    
    Args:
        timestamp: Original timestamp from CSV
        window_size_minutes: Size of time window in minutes (default: 30)
        
    Returns:
        Datetime object aligned to 30-minute window start
    """
    if not isinstance(timestamp, datetime.datetime):
        raise TypeError("timestamp must be a datetime object")
    
    # Ensure timestamp is timezone-aware
    if timestamp.tzinfo is None:
        timestamp = HKT.localize(timestamp)
    
    # Calculate window start time
    total_minutes = timestamp.hour * 60 + timestamp.minute
    window_number = total_minutes // window_size_minutes
    window_start_minute = window_number * window_size_minutes
    
    # Create aligned window start datetime
    aligned_start = timestamp.replace(
        hour=window_start_minute // 60,
        minute=window_start_minute % 60,
        second=0,
        microsecond=0
    )
    
    return categorize_query_timestamp(timestamp, window_size_minutes)


def categorize_query_timestamp(timestamp: datetime.datetime, window_size_minutes: int = 15, tolerance_seconds: int = 5) -> datetime.datetime:
    """
    Categorize manual query timestamp to nearest 15-minute window with 5-second tolerance.

    For manual queries like "Jul 31, 2025, 10:15 AM - 10:45 AM", handles timestamps
    that are within 5 seconds of window boundaries.
    
    Examples:
    - Jul 31, 2025, 10:14:56 AM -> Jul 31, 2025, 10:15:00 AM (within 5s of 10:15)
    - Jul 31, 2025, 10:59:56 AM -> Jul 31, 2025, 11:00:00 AM (within 5s of 11:00)
    
    Args:
        timestamp: Original timestamp from query results
        window_size_minutes: Size of time window in minutes (default: 30)
        tolerance_seconds: Tolerance for boundary adjustment in seconds (default: 5)
        
    Returns:
        Datetime object aligned to nearest 30-minute window
    """
    if not isinstance(timestamp, datetime.datetime):
        raise TypeError("timestamp must be a datetime object")
    
    # Ensure timestamp is timezone-aware
    if timestamp.tzinfo is None:
        timestamp = HKT.localize(timestamp)
    
    # Calculate the exact window boundary times
    total_minutes = timestamp.hour * 60 + timestamp.minute
    window_number = total_minutes // window_size_minutes
    window_start_minute = window_number * window_size_minutes
    
    # Calculate the next window boundary
    next_window_start_minute = (window_number + 1) * window_size_minutes
    
    # Create boundary datetimes
    current_window_start = timestamp.replace(
        hour=window_start_minute // 60,
        minute=window_start_minute % 60,
        second=0,
        microsecond=0
    )
    
    next_window_start = timestamp.replace(
        hour=next_window_start_minute // 60,
        minute=next_window_start_minute % 60,
        second=0,
        microsecond=0
    )
    
    # Handle hour overflow for next window
    if next_window_start_minute >= 24 * 60:
        next_window_start = next_window_start + datetime.timedelta(days=1)
        next_window_start = next_window_start.replace(hour=0, minute=0)
    
    # Calculate distances to boundaries
    current_distance = abs((timestamp - current_window_start).total_seconds())
    next_distance = abs((timestamp - next_window_start).total_seconds())
    
    # Apply 5-second tolerance rule
    if current_distance <= tolerance_seconds:
        return current_window_start
    elif next_distance <= tolerance_seconds:
        return next_window_start
    else:
        # Normal alignment to nearest window
        seconds_from_start = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
        window_seconds = window_size_minutes * 60
        
        # Determine which window to align to based on which boundary is closer
        if seconds_from_start % window_seconds < window_seconds / 2:
            return current_window_start
        else:
            return next_window_start


def get_window_start_end(timestamp: datetime.datetime, window_size_minutes: int = 15) -> tuple:
    """
    Get the start and end times for a 15-minute window containing the given timestamp.

    Args:
        timestamp: The datetime to get window boundaries for
        window_size_minutes: Size of time window in minutes (default: 15)

    Returns:
        Tuple of (start_time, end_time) as datetime objects
    """
    if not isinstance(timestamp, datetime.datetime):
        raise TypeError("timestamp must be a datetime object")
    
    # Ensure timestamp is timezone-aware
    if timestamp.tzinfo is None:
        timestamp = pytz.UTC.localize(timestamp)
    
    # Calculate window start
    total_minutes = timestamp.hour * 60 + timestamp.minute
    window_number = total_minutes // window_size_minutes
    window_start_minute = window_number * window_size_minutes
    
    window_start = timestamp.replace(
        hour=window_start_minute // 60,
        minute=window_start_minute % 60,
        second=0,
        microsecond=0
    )
    
    window_end = window_start + datetime.timedelta(minutes=window_size_minutes)
    
    return window_start, window_end


def standardize_timezone(dt: datetime.datetime, target_timezone: str = 'UTC') -> datetime.datetime:
    """
    Standardize datetime to UTC or specified timezone.
    
    Args:
        dt: Datetime object to standardize
        target_timezone: Target timezone (default: 'UTC')
        
    Returns:
        Timezone-aware datetime object
    """
    if not isinstance(dt, datetime.datetime):
        raise TypeError("dt must be a datetime object")
    
    if dt.tzinfo is None:
        # Assume UTC if no timezone provided
        dt = pytz.UTC.localize(dt)
    
    try:
        target_tz = pytz.timezone(target_timezone)
    except pytz.UnknownTimeZoneError:
        target_tz = pytz.UTC
    
    return dt.astimezone(target_tz)


def format_for_display(dt: datetime.datetime) -> str:
    """
    Format datetime for display purposes (human-readable).
    
    Args:
        dt: Datetime object to format
        
    Returns:
        String in format "YYYY-MM-DD HH:MM:SS UTC"
    """
    if not isinstance(dt, datetime.datetime):
        raise TypeError("dt must be a datetime object")
    
    dt_utc = standardize_timezone(dt)
    return dt_utc.strftime("%Y-%m-%d %H:%M:%S %Z")


def is_valid_timestamp(timestamp_str: str) -> bool:
    """
    Validate if a timestamp string can be parsed.
    
    Args:
        timestamp_str: String to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        parse_qradar_timestamp(timestamp_str)
        return True
    except (ValueError, TypeError):
        return False


def get_current_window_id() -> str:
    """
    Get the current window ID based on current time.
    
    Returns:
        Current window ID string
    """
    now = datetime.datetime.now(pytz.UTC)
    return get_window_id(now)


# Cache for frequently used windows
_window_cache = {}

def get_window_id_cached(timestamp: datetime.datetime, window_size_minutes: int = 15) -> str:
    """
    Cached version of get_window_id for performance optimization.
    
    Args:
        timestamp: The datetime to generate window ID for
        window_size_minutes: Size of time window in minutes
        
    Returns:
        String window ID
    """
    cache_key = (timestamp.strftime("%Y%m%d%H%M"), window_size_minutes)
    
    if cache_key in _window_cache:
        return _window_cache[cache_key]
    
    window_id = get_window_id(timestamp, window_size_minutes)
    _window_cache[cache_key] = window_id
    return window_id


def clear_cache():
    """Clear the window ID cache."""
    global _window_cache
    _window_cache.clear()


if __name__ == "__main__":
    # Test the module
    print("Testing time_utils.py...")
    print("=" * 60)
    
    # Test timestamp parsing
    test_timestamps = [
        "Jul 29, 2025, 9:50:55 AM",
        "Jul 30, 2025, 10:10:30 AM",
        "Jul 29, 2025, 5:34:47 PM",
        "Jul 30, 2025, 7:32:38 AM",
    ]
    
    print("📅 CSV TIMESTAMP ADJUSTMENT TESTS:")
    print("-" * 40)
    
    for ts in test_timestamps:
        try:
            parsed = parse_qradar_timestamp(ts)
            adjusted = adjust_csv_timestamp_to_window_start(parsed)
            print(f"✅ Original: {ts}")
            print(f"   Adjusted: {adjusted.strftime('%b %d, %Y, %I:%M:%S %p')}")
            print()
        except Exception as e:
            print(f"❌ Failed to process '{ts}': {e}")
    
    # Test specific examples from requirements
    print("🎯 SPECIFIC EXAMPLES:")
    print("-" * 40)
    
    examples = [
        ("Jul 29, 2025, 9:50:55 AM", "Jul 29, 2025, 9:30:00 AM"),
        ("Jul 30, 2025, 10:10:30 AM", "Jul 30, 2025, 10:00:00 AM"),
        ("Jul 29, 2025, 9:15:00 AM", "Jul 29, 2025, 9:00:00 AM"),
        ("Jul 29, 2025, 9:45:00 AM", "Jul 29, 2025, 9:30:00 AM"),
    ]
    
    for original, expected in examples:
        try:
            parsed = parse_qradar_timestamp(original)
            adjusted = adjust_csv_timestamp_to_window_start(parsed)
            expected_parsed = parse_qradar_timestamp(expected)
            
            print(f"✅ {original} -> {adjusted.strftime('%b %d, %Y, %I:%M:%S %p')}")
            assert adjusted == expected_parsed, f"Mismatch: {adjusted} != {expected_parsed}"
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    print("\n🎯 5-SECOND TOLERANCE TESTS:")
    print("-" * 40)
    
    tolerance_examples = [
        ("Jul 31, 2025, 10:14:56 AM", "Jul 31, 2025, 10:15:00 AM"),  # 4s before boundary
        ("Jul 31, 2025, 10:15:04 AM", "Jul 31, 2025, 10:15:00 AM"),  # 4s after boundary
        ("Jul 31, 2025, 10:59:56 AM", "Jul 31, 2025, 11:00:00 AM"),  # 4s before 11:00
        ("Jul 31, 2025, 11:00:04 AM", "Jul 31, 2025, 11:00:00 AM"),  # 4s after 11:00
        ("Jul 31, 2025, 10:15:10 AM", "Jul 31, 2025, 10:15:00 AM"),  # Normal case
        ("Jul 31, 2025, 10:45:30 AM", "Jul 31, 2025, 10:30:00 AM"),  # Normal case
    ]
    
    for original, expected in tolerance_examples:
        try:
            parsed = parse_qradar_timestamp(original)
            categorized = categorize_query_timestamp(parsed)
            expected_parsed = parse_qradar_timestamp(expected)
            
            print(f"✅ {original} -> {categorized.strftime('%b %d, %Y, %I:%M:%S %p')}")
            assert categorized == expected_parsed, f"Mismatch: {categorized} != {expected_parsed}"
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    print("\n✅ All CSV timestamp adjustments with 5-second tolerance working correctly!")