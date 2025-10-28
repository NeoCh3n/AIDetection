import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from system import logging_utils
import requests
import urllib3
import time
import threading
import random
from typing import Optional
urllib3.disable_warnings()

# Module-wide lock and timestamp to throttle QRadar result retrievals
_RESULT_THROTTLE_LOCK = threading.Lock()
_LAST_RESULT_ATTEMPT_EPOCH: float = 0.0


def _enforce_result_throttle(min_interval_seconds: float, reason: Optional[str] = None) -> float:
    """
    Enforce a minimum delay between successive QRadar result fetches.

    Args:
        min_interval_seconds: Minimum spacing between API calls in seconds.
        reason: Optional string appended to the informational log message.

    Returns:
        float: Seconds slept to satisfy the throttle (0 if none).
    """
    if min_interval_seconds <= 0:
        return 0.0

    wait_seconds = 0.0
    with _RESULT_THROTTLE_LOCK:
        global _LAST_RESULT_ATTEMPT_EPOCH
        now = time.time()
        elapsed = now - _LAST_RESULT_ATTEMPT_EPOCH
        if elapsed < min_interval_seconds:
            wait_seconds = min_interval_seconds - elapsed
            _LAST_RESULT_ATTEMPT_EPOCH = now + wait_seconds
        else:
            _LAST_RESULT_ATTEMPT_EPOCH = now

    if wait_seconds > 0:
        msg = f"Throttling QRadar result retrieval for {wait_seconds:.2f}s"
        if reason:
            msg = f"{msg} ({reason})"
        logging_utils.run_log("INFO", msg)
        time.sleep(wait_seconds)

    with _RESULT_THROTTLE_LOCK:
        _LAST_RESULT_ATTEMPT_EPOCH = time.time()

    return wait_seconds


def _compute_retry_delay(
    attempt_number: int,
    base_delay: float,
    max_delay: float,
    retry_after_header: Optional[str] = None,
    jitter: float = 1.0,
) -> float:
    """
    Calculate delay before retrying a result fetch.

    Args:
        attempt_number: Current attempt count (1-indexed).
        base_delay: Minimum delay for exponential backoff.
        max_delay: Maximum delay permitted between retries.
        retry_after_header: Optional Retry-After header value in seconds.
        jitter: Random jitter upper bound to de-synchronize concurrent callers.

    Returns:
        float: Delay in seconds before next retry.
    """
    delay = base_delay * (2 ** (attempt_number - 1))
    delay = min(max_delay, max(base_delay, delay))

    if retry_after_header:
        try:
            delay = max(delay, float(retry_after_header))
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
    'Connection': 'close'
}

Qradar_address_default = "192.168.153.123"
search_id_default = "6c1b5627-e9f1-45a9-9040-7bab65a6463b"


def result_searches_Qradar(
    Qradar_address=Qradar_address_default,
    search_id=search_id_default,
    request_header=request_header_default,
    timeout: int = 60,
    min_interval_seconds: float = 10.0,
    max_retries: int = 4,
    backoff_base_seconds: float = 5.0,
    backoff_max_seconds: float = 60.0,
    jitter_seconds: float = 1.0,
):
    """
    Retrieve QRadar Ariel search results with throttling and retry backoff.

    Args:
        Qradar_address: Target QRadar host.
        search_id: Identifier returned from the search creation endpoint.
        request_header: Headers containing security token and content settings.
        timeout: Per-request timeout in seconds.
        min_interval_seconds: Minimum spacing enforced between API calls.
        max_retries: Maximum number of attempts before giving up.
        backoff_base_seconds: Base delay used when computing exponential backoff.
        backoff_max_seconds: Maximum allowable backoff between retries.
        jitter_seconds: Random jitter upper bound to prevent synchronized retries.

    Returns:
        dict or None: Parsed JSON payload containing search results on success; otherwise None.
    """
    request_URI = f"https://{Qradar_address}/api/ariel/searches/{search_id}/results"
    attempts = 0
    last_error: Optional[str] = None

    while attempts < max_retries:
        attempts += 1
        retry_after_header: Optional[str] = None
        _enforce_result_throttle(
            min_interval_seconds,
            reason=f"attempt {attempts}/{max_retries}",
        )

        try:
            response = requests.get(
                request_URI,
                headers=request_header,
                verify=False,
                timeout=timeout,
            )
            logging_utils.run_log(
                "INFO",
                f"QRadar result GET sent (attempt {attempts}/{max_retries}); status={response.status_code}",
            )
        except requests.Timeout as exc:
            last_error = f"Timeout after {timeout}s: {exc}"
            logging_utils.run_log(
                "WARNING",
                f"QRadar result retrieval timeout (attempt {attempts}/{max_retries}): {exc}",
            )
            response = None
        except requests.RequestException as exc:
            last_error = str(exc)
            logging_utils.run_log(
                "WARNING",
                f"QRadar result retrieval request error (attempt {attempts}/{max_retries}): {exc}",
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

            if response.ok and isinstance(response_payload, dict):
                events = response_payload.get('events')
                if isinstance(events, list):
                    record_count = len(events)
                    logging_utils.run_log(
                        "INFO",
                        f"QRadar result retrieval succeeded with {record_count} record(s) (attempt {attempts}/{max_retries})",
                    )
                    return response_payload

            status_code = response.status_code
            body_preview = response_payload if response_payload is not None else response.text
            last_error = f"Unexpected response (status={status_code}): {body_preview}"
            logging_utils.run_log(
                "WARNING",
                f"QRadar result retrieval unexpected response (attempt {attempts}/{max_retries}): status={status_code} body={body_preview}",
            )

            should_retry = (
                attempts < max_retries
                and (
                    status_code in (202, 204, 429)
                    or 500 <= status_code < 600
                )
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
            f"Retrying QRadar result retrieval in {delay:.2f}s (attempt {attempts + 1}/{max_retries})",
        )
        time.sleep(delay)

    logging_utils.run_log(
        "ERROR",
        f"QRadar result retrieval failed after {attempts} attempt(s): {last_error or 'Unknown error'}",
    )
    return None

#### default
if __name__ == "__main__":
    get_response_ariel_searches_results = result_searches_Qradar(Qradar_address = Qradar_address_default , search_id = search_id_default , request_header = request_header_default )
    if get_response_ariel_searches_results and "events" in get_response_ariel_searches_results:
        print("Number of items:", len(get_response_ariel_searches_results["events"]))
    else:
        print("Error: No results returned or unexpected response format")
        print(get_response_ariel_searches_results)
