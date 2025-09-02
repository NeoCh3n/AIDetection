#!/usr/bin/env python3
"""
Smoke test for logging to running_log.

This test exercises system.logging_utils to ensure logs are written to
the configured running_log directory and prints the path of the log file
created so the caller can inspect it.
"""

import os
import sys
from datetime import datetime

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system import logging_utils  # type: ignore
from system import config  # type: ignore


def main() -> None:
    # Emit a few log lines
    logging_utils.run_log("INFO", "Test: run_log basic entry")
    logging_utils.log_pipeline_event("TEST", "Test: pipeline event", {"module": "test_logging_running_log"})
    logging_utils.log_detection(
        hostname="test-host",
        window_id=datetime.now().strftime("%Y-%m-%d_%H-%M"),
        prediction="normal",
        confidence=0.12,
    )

    # Determine today's log file path
    log_dir = config.log_dir_path_default
    # logging_utils names files by (now - 15 minutes).date()
    today_name = str((datetime.now()).date())
    log_file = os.path.join(log_dir, f"{today_name}.log")

    print("\n--- Logging Smoke Test ---")
    print(f"Log directory: {os.path.abspath(log_dir)}")
    print(f"Expected log file: {os.path.abspath(log_file)}")
    if os.path.exists(log_file):
        print("Log file exists. Last 10 lines:")
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()[-10:]
            for line in lines:
                sys.stdout.write(line)
        except Exception as e:
            print(f"(Could not read log file: {e})")
    else:
        print("Log file not found yet. Check running_log directory.")


if __name__ == "__main__":
    main()

