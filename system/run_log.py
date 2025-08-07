import time
import datetime
import os
import send_syslog
import config

fetch_data_frequency = config.fetch_data_frequency_default
log_dir_path = config.log_dir_path_default
log_destination_address = config.log_destination_address_default
log_destination_port = config.log_destination_port_default

def run_log(level, message, payload = None):
    date_lastweek = str(datetime.datetime.now() - datetime.timedelta(days=7))[:10]
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
        log_message = "| " + level.ljust(10) + " | " + str(time_now) + " | " + "Session: " + str(session_count) + "/" + str(total_session) + " | " + message + "\n"
        if payload is not None:
            log_message += " | Payload: " + str(payload) + "\n"
        logfile_today.write(log_message)
        print(log_message)

    # send the logs to Qradar
    if level == "AIDA-ML":
        send_syslog.send_syslog(log_destination_address, log_destination_port, "AIDA-ML", message)
    else:
        send_syslog.send_syslog(log_destination_address, log_destination_port, "AIDA-LOG", log_message)

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
