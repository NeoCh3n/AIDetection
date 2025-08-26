import time
import datetime
import os
from pathlib import Path
from . import send_syslog
from . import config

fetch_data_frequency = config.fetch_data_frequency_default
log_dir_path = config.log_dir_path_default
log_destination_address = config.log_destination_address_default
log_destination_port = config.log_destination_port_default
data_retention_days = getattr(config, "DATA_RETENTION_DAYS", 7)
header_ml = getattr(config, "SYSLOG_HEADER_ML", "AI+xx-ML")
header_log = getattr(config, "SYSLOG_HEADER_LOG", "AI+xx-LOG")

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
        print(log_message)

    # send the logs to Qradar
    # Sanitize syslog message to a single line and cap size
    def _sanitize(msg: str, max_len: int = 8192) -> str:
        msg1 = msg.replace("\n", " ").replace("\r", " ")
        return msg1[:max_len]

    try:
        if level == header_ml:
            send_syslog.send_syslog(log_destination_address, log_destination_port, header_ml, _sanitize(message))
        else:
            send_syslog.send_syslog(log_destination_address, log_destination_port, header_log, _sanitize(log_message))
    except Exception as e:
        # Fallback: write a local error note
        with open(filepath_today, "a+") as logfile_today:
            logfile_today.write(f"| {'ERROR'.ljust(10)} | {time_now} | Syslog send failed: {e}\n")

if __name__ == "__main__":
    run_log("DEBUG", "debug log")
    time.sleep(5)
    run_log("INFO", "info log")
    time.sleep(5)
    run_log("WARNING", "warning log")
    time.sleep(5)
    run_log("ERROR", "error log")
    time.sleep(5)
    run_log("CRITICAL", "critical log")
    time.sleep(5)
