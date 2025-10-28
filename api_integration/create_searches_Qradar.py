import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from system import logging_utils
import requests
from urllib.parse import urlencode
import urllib3
import time
import threading
import random
from typing import Optional
urllib3.disable_warnings()

# Module-wide lock and timestamp to throttle QRadar search creation
_SEARCH_THROTTLE_LOCK = threading.Lock()
_LAST_SEARCH_ATTEMPT_EPOCH: float = 0.0


def _enforce_search_throttle(min_interval_seconds: float, reason: Optional[str] = None) -> float:
    """
    Ensure a minimum delay between successive QRadar search creations.

    Args:
        min_interval_seconds: Minimum interval that should separate POST attempts.
        reason: Optional reason used in log messages.

    Returns:
        float: The number of seconds slept to enforce the throttle.
    """
    if min_interval_seconds <= 0:
        return 0.0

    wait_seconds = 0.0
    with _SEARCH_THROTTLE_LOCK:
        global _LAST_SEARCH_ATTEMPT_EPOCH
        now = time.time()
        elapsed = now - _LAST_SEARCH_ATTEMPT_EPOCH
        if elapsed < min_interval_seconds:
            wait_seconds = min_interval_seconds - elapsed
            # Block additional attempts until we actually wait
            _LAST_SEARCH_ATTEMPT_EPOCH = now + wait_seconds
        else:
            _LAST_SEARCH_ATTEMPT_EPOCH = now

    if wait_seconds > 0:
        msg = f"Throttling QRadar search creation for {wait_seconds:.2f}s"
        if reason:
            msg = f"{msg} ({reason})"
        logging_utils.run_log("INFO", msg)
        time.sleep(wait_seconds)

    with _SEARCH_THROTTLE_LOCK:
        _LAST_SEARCH_ATTEMPT_EPOCH = time.time()

    return wait_seconds


def _compute_retry_delay(
    attempt_number: int,
    base_delay: float,
    max_delay: float,
    retry_after_header: Optional[str] = None,
    jitter: float = 1.0,
) -> float:
    """
    Calculate the delay before the next retry attempt.

    Args:
        attempt_number: Current attempt number (1-indexed).
        base_delay: Base delay used for exponential backoff.
        max_delay: Maximum allowable delay.
        retry_after_header: Optional Retry-After header value (seconds).
        jitter: Maximum random jitter to add (uniform in [0, jitter]).

    Returns:
        float: Delay in seconds before the next retry.
    """
    delay = base_delay * (2 ** (attempt_number - 1))
    delay = min(max_delay, max(base_delay, delay))

    if retry_after_header:
        try:
            retry_after_value = float(retry_after_header)
            delay = max(delay, retry_after_value)
        except (TypeError, ValueError):
            pass

    if jitter > 0:
        delay += random.uniform(0, jitter)

    return delay

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

def create_searches_Qradar(
    qradar_address=Qradar_address_default,
    AQL=AQL_default,
    request_header=request_header_default,
    timeout: int = 60,
    min_interval_seconds: float = 15.0,
    max_retries: int = 3,
    backoff_base_seconds: float = 5.0,
    backoff_max_seconds: float = 60.0,
    jitter_seconds: float = 2.0,
):
    """
    Create an Ariel search in QRadar with built-in throttling and retry backoff.

    Args:
        qradar_address: QRadar host address.
        AQL: Query string sent to QRadar.
        request_header: Headers containing authentication/token information.
        timeout: Per-request timeout in seconds.
        min_interval_seconds: Minimum time gap enforced between requests.
        max_retries: Maximum number of attempts (initial + retries).
        backoff_base_seconds: Base delay used for exponential backoff.
        backoff_max_seconds: Maximum delay allowed between retries.
        jitter_seconds: Random jitter added to backoff to avoid thundering herd.

    Returns:
        dict or None: Successful QRadar response payload containing the search_id, or None on failure.
    """
    base_uri = "https://" + qradar_address + "/api/ariel/searches"
    attempts = 0
    last_error: Optional[str] = None

    while attempts < max_retries:
        attempts += 1
        retry_after_header: Optional[str] = None
        _enforce_search_throttle(
            min_interval_seconds,
            reason=f"attempt {attempts}/{max_retries}",
        )

        try:
            response = requests.post(
                base_uri,
                headers=request_header,
                params={"query_expression": AQL},
                verify=False,
                timeout=timeout,
            )
            logging_utils.run_log(
                "INFO",
                f"QRadar search POST sent (attempt {attempts}/{max_retries}); status={response.status_code}",
            )
        except requests.Timeout as exc:
            last_error = f"Timeout after {timeout}s: {exc}"
            logging_utils.run_log(
                "WARNING",
                f"QRadar create_search timeout (attempt {attempts}/{max_retries}): {exc}",
            )
            response = None
        except requests.RequestException as exc:
            last_error = str(exc)
            logging_utils.run_log(
                "WARNING",
                f"QRadar create_search request error (attempt {attempts}/{max_retries}): {exc}",
            )
            response = None

        if response is None:
            should_retry = attempts < max_retries
        else:
            retry_after_header = response.headers.get("Retry-After")
            try:
                response_payload = response.json()
            except ValueError:
                response_payload = None

            if response.ok and isinstance(response_payload, dict) and "search_id" in response_payload:
                search_id = response_payload["search_id"]
                logging_utils.run_log(
                    "INFO",
                    f"QRadar search created successfully (search_id={search_id}) on attempt {attempts}/{max_retries}",
                )
                return response_payload

            status_code = response.status_code
            body_preview = response_payload if response_payload is not None else response.text
            last_error = f"Unexpected response (status={status_code}): {body_preview}"
            logging_utils.run_log(
                "WARNING",
                f"QRadar create_search unexpected response (attempt {attempts}/{max_retries}): status={status_code} body={body_preview}",
            )
            should_retry = (
                attempts < max_retries
                and (status_code == 429 or 500 <= status_code < 600)
            )

        if not should_retry:
            break

        delay = _compute_retry_delay(
            attempts,
            backoff_base_seconds,
            backoff_max_seconds,
            retry_after_header=retry_after_header if response is not None else None,
            jitter=jitter_seconds,
        )
        logging_utils.run_log(
            "INFO",
            f"Retrying QRadar search creation in {delay:.2f}s (attempt {attempts + 1}/{max_retries})",
        )
        time.sleep(delay)

    logging_utils.run_log(
        "ERROR",
        f"QRadar create_search failed after {attempts} attempt(s): {last_error or 'Unknown error'}",
    )
    return None

if __name__ == "__main__":
    result = create_searches_Qradar(qradar_address = Qradar_address_default, AQL = AQL_default, request_header = request_header_default)
    if result and "search_id" in result:
        search_id = result["search_id"]
        print(search_id)
    else:
        print("Error: Failed to create search or no search_id returned")
