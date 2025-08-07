# ############### #
# ############### #
#### Environment
import create_searches_Qradar
import status_searches_Qradar
import result_searches_Qradar
import delete_searches_Qradar

import send_syslog

import get_DB
import insert_DB
import query_DB
import delete_DB

import calculate_stats
import other_functions
import run_log

import config

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import joblib
import syslog
import urllib3
urllib3.disable_warnings()
import pandas as pd
import numpy as np
import statistics
import csv
import math
from collections import defaultdict, Counter
import datetime
import time
import json

###################

#### Qradar configuration
Qradar_address = config.Qradar_address_default
Qradar_token = config.Qradar_token_default

request_header = config.request_header_default

#############################
#### MongoDB configuration
# CONNECTION STRINGS
CONNECTION_STRING = config.MONGODB_CONNECTION_STRING

# DB NAM
DB_NAME = config.MONGODB_DB_NAME

# COLLECTION NAME
COLLECTION_NAME = config.MONGODB_COLLECTION_NAME

#### Change AQL here
AQL = config.AQL_default

##################
# ############## #
#### Parameters
# Frequency of fetching data from Qradar (every _ minutes)
# If you change fetch_data_frequency_default, please also edit the schedule task in crontab
fetch_data_frequency = config.fetch_data_frequency_default

# Analyze time (every _ minutes)
# Please do not change
# If you change analyze_duration_default, please train and apply a new model
analyze_duration = config.analyze_duration_default

# Maximum record count
maximum_record_count = config.maximum_record_count_default

model_path = config.model_path_default

# Use new model paths from config
svm_model_path = config.svm_model_path_default
logreg_model_path = config.logreg_model_path_default
rf_model_path = config.rf_model_path_default
voting_model_path = config.voting_model_path_default


def main():
    run_log.run_log("INFO", "1. The program is started")

    # Load all three models using config paths
    svm_model_hostname = joblib.load(svm_model_path)
    logreg_model_hostname = joblib.load(logreg_model_path)
    rf_model_hostname = joblib.load(rf_model_path)
    voting_model_hostname = joblib.load(voting_model_path)

    #### Delete data from MongoDB
    date_lastweek = str(datetime.datetime.now() - datetime.timedelta(days=7))[:10]
    query_DB_lastweek = {'Date': date_lastweek}
    deleted_data_count = delete_DB.delete_database(CONNECTION_STRING, "SIEM", "NetworkConnection", query_DB_lastweek)

    #### Fetch Data from Qradar
    date_now = str(datetime.datetime.now())[:10]
    time_now = str(datetime.datetime.now())[11:16]

    date_15mins_ago = str(datetime.datetime.now() - datetime.timedelta(minutes=fetch_data_frequency))[:10]
    time_15mins_ago = str(datetime.datetime.now() - datetime.timedelta(minutes=fetch_data_frequency))[11:16]

    date_30mins_ago = str(datetime.datetime.now() - datetime.timedelta(minutes=analyze_duration))[:10]
    time_30mins_ago = str(datetime.datetime.now() - datetime.timedelta(minutes=analyze_duration))[11:16]

    POST_Request_ariel_searches_STARTDATE = date_15mins_ago
    POST_Request_ariel_searches_ENDDATE = date_now
    POST_Request_ariel_searches_starttime = time_15mins_ago
    POST_Request_ariel_searches_endtime = time_now

    #### POST REQUEST
    ariel_search_id = create_searches_Qradar.create_searches_Qradar(Qradar_address, AQL, request_header)["search_id"]
    if ariel_search_id:
        time.sleep(fetch_data_frequency)
    else:
        return

    #### check if the search is completed
    for i in range(fetch_data_frequency):
        run_log.run_log("INFO", "4. Trial #" + str(i + 1) + " of sending GET request..")
        #### GET REQUEST
        search_info = status_searches_Qradar.status_searches_Qradar(Qradar_address, ariel_search_id, request_header)
        search_status = search_info["status"]
        search_record_count = int(search_info["record_count"])

        if search_status == "EXECUTE":
            time.sleep(60)

        elif search_status == "COMPLETED" and search_record_count < maximum_record_count:
            #### GET ariel searches result
            ariel_searches_results = result_searches_Qradar.result_searches_Qradar(Qradar_address, ariel_search_id, request_header)

            if 'events' in ariel_searches_results.keys():
                #### Adding header to data
                ariel_searches_results['Date'] = POST_Request_ariel_searches_STARTDATE
                ariel_searches_results['Time_Interval'] = POST_Request_ariel_searches_starttime + "->" + POST_Request_ariel_searches_endtime

                for eachrow in ariel_searches_results['events']:
                    eachrow_time = datetime.datetime.fromtimestamp(eachrow['Log Source Time'] / 1000)
                    eachrow['Date'] = eachrow_time.strftime("%Y-%m-%d")
                    eachrow['Time'] = eachrow_time.strftime("%H:%M:%S")  

                #### Insert ariel searches results to MongoDB
                insert_DB.insert_database(CONNECTION_STRING, "SIEM", "NetworkConnection", ariel_searches_results)

                #### Delete ariel search after complete
                #### DELETE REQUEST
                delete_response = delete_searches_Qradar.delete_searches_Qradar(Qradar_address, ariel_search_id, request_header)
                break

        else:
            run_log.run_log("ERROR", "Invalid ariel search" + str(ariel_searches_results))
            return

    else:  # This else corresponds to the for loop finishing without break
        run_log.run_log("ERROR", "Failed to check the search status -- status: " + search_status + " , info: " + str(search_info))
        return

    # ############################## #
    #### Query MongoDB to get data
    # ############################## #
    ### DO calculation every 30 mins
    # Query data from MongoDB

    data = []

    get_dot_DATE = POST_Request_ariel_searches_STARTDATE  # Assuming this date is correct context
    get_dot_STARTTIME = time_30mins_ago  # Assuming this time is correct context
    get_dot_ENDTIME = POST_Request_ariel_searches_endtime  # Assuming this time is correct context

    #define a standard time interval
    analysis_time_interval = get_dot_STARTTIME + "->" + get_dot_ENDTIME

    #### Query data from MongoDB
    MongoDB_Query = {"Date": {"$gte": get_dot_DATE}}  # Simplified based on image, might need refinement
    data_SIEM_NetworkConnection = query_DB.query_database(CONNECTION_STRING, "SIEM", "NetworkConnection", MongoDB_Query)

    for eachcsv in data_SIEM_NetworkConnection:
        #### get data by time interval
        if eachcsv['Date'] >= get_dot_DATE:
            idx = eachcsv["Time_Interval"].index("->")
            data_starttime = datetime.datetime.strptime(eachcsv["Time_Interval"][0:idx], "%H:%M")
            data_endtime = datetime.datetime.strptime(eachcsv["Time_Interval"][idx + len("->"):], "%H:%M")  # Corrected index based on pattern
            query_starttime = datetime.datetime.strptime(get_dot_STARTTIME, "%H:%M")
            """query_endtime = datetime.datetime.strptime(get_dot_ENDTIME, "%H:%M")

            if data_starttime >= query_starttime and data_endtime <= query_endtime:
                for eachrow in eachcsv['events']:
                    data.append(eachrow)"""
            query_endtime = datetime.datetime.strptime(get_dot_ENDTIME, "%H:%M")

            if data_starttime >= query_starttime and data_endtime <= query_endtime:
                # 从批次文档中获取 Time_Interval
                time_interval_for_events = eachcsv.get("Time_Interval")
                for eachrow in eachcsv['events']:
                    # 将 Time_Interval 添加到每个事件记录中
                    #eachrow['Time_interval'] = time_interval_for_events
                    eachrow['Time_interval'] = analysis_time_interval
                    data.append(eachrow)


    run_log.run_log("INFO", "13. Total number of data in " + get_dot_STARTTIME + "->" + get_dot_ENDTIME + " = " + str(len(data)))
        


    ##############################
    #### Analyzing data with ML
    ##############################
    #### group the data
    alert_count = 0
    hostname_count = 0

    run_log.run_log("INFO", "14. Analyzing the data with ML")

    #### Count process path occurrences
    """process_path_counts = Counter()
    for event in data:
        if 'Process Path (custom)' in event:
            process_path_counts[event['Process Path (custom)']] += 1
            #print(event['Process Path (custom)'])"""
    process_path_counts = Counter(
    (event['Date'], event['Time_interval'], event['sysmon_hostname (custom)'], event['Process Path (custom)']) for event in data if 'Process Path (custom)' in event and 'sysmon_hostname (custom)' in event)
    
    """#### group by DATE
    grouped_DATE = other_functions.groupby(data, 'Date')
    for each_date in grouped_DATE:

        #### group by Time_interval
        grouped_by_interval = other_functions.groupby(events_by_date, 'Time_interval')
        for each_interval, events_by_interval in grouped_by_interval.items():

            #### group by HOSTNAME
            grouped_DATE_HOSTNAME = other_functions.groupby(grouped_DATE[each_date], 'sysmon_hostname (custom)')  # Check actual field name
            for each_date_hostname in grouped_DATE_HOSTNAME:
                hostname_count += 1

                #### group by path
                grouped_DATE_HOSTNAME_PATH = other_functions.groupby(grouped_DATE_HOSTNAME[each_date_hostname], 'Process Path (custom)')  # Check actual field name
                for each_date_hostname_path in grouped_DATE_HOSTNAME_PATH:

                    #### Calculate dots grouped by path
                    if each_date == get_dot_DATE:

                        timestamps = []
                        #### count here
                        for each_date_hostname_path_time in grouped_DATE_HOSTNAME_PATH[each_date_hostname_path]:
                            time_str = each_date_hostname_path_time['Time']
                            timestamps.append(time_str)

                        # Get the count of this process path in the entire dataset (new feature)
                        current_process_path = each_date_hostname_path_time['Process Path (custom)']
                        process_path_occurence_count = process_path_counts[current_process_path]
                        
                        #print("current process path:", current_process_path)
                        #print("process_path_occurrence_count:", process_path_counts[current_process_path])

                        if len(timestamps)> fetch_data_frequency:
                            stats = calculate_stats.calculate_stats(timestamps, process_path_occurence_count)

                            if stats[0]>1 and stats[1]>0 and stats[2] != False and stats[3] >1:
                                # Predict with all three models
                                svm_result = svm_model_hostname.predict([[stats[0], stats[1], stats[2], stats[3]]])
                                svm_proba = svm_model_hostname.predict_proba([[stats[0], stats[1], stats[2], stats[3]]])[0][1]
                                logreg_result = logreg_model_hostname.predict([[stats[0], stats[1], stats[2], stats[3]]])
                                logreg_proba = logreg_model_hostname.predict_proba([[stats[0], stats[1], stats[2], stats[3]]])[0][1]
                                rf_result = rf_model_hostname.predict([[stats[0], stats[1], stats[2], stats[3]]])
                                rf_proba = rf_model_hostname.predict_proba([[stats[0], stats[1], stats[2], stats[3]]])[0][1]

                                # Voting model prediction
                                voting_result = voting_model_hostname.predict([[stats[0], stats[1], stats[2], stats[3]]])
                                voting_proba = voting_model_hostname.predict_proba([[stats[0], stats[1], stats[2], stats[3]]])[0][1]
                                #print("current_process_path:", current_process_path)
                                #print(stats)
                                #print("svm_result:",svm_result,"svm_proba:",svm_proba, "logreg_result:",logreg_result,"logreg_proba:",logreg_proba,"rf_result:",rf_result,"rf_proba:",rf_proba)

                                # Alert if any model detects attack (remove 0.8 probability threshold)
                                if (svm_result[0] == 1) or (logreg_result[0] == 1) or (rf_result[0] == 1) or (voting_result[0] == 1):
                                    dots_row = {}
                                    dots_row['Alert_ID'] = 2
                                    dots_row['Alert_Description'] = "Suspect C2 Traffic"
                                    dots_row['Confident_Level'] = str(float(round(np.mean([svm_proba, logreg_proba, rf_proba, voting_proba]) * 100, 2))) + "%"
                                    dots_row['Date'] = each_date_hostname_path_time['Date']
                                    dots_row['Time'] = get_dot_STARTTIME + "-" + get_dot_ENDTIME
                                    dots_row['System_Hostname'] = each_date_hostname
                                    dots_row['Source IP'] = each_date_hostname_path_time['Source IP']
                                    # Fix string escape for backslash
                                    dots_row['Process_Path'] = str(each_date_hostname_path).replace('\\\\', '\\')
                                    dots_row['Runtime'] = round(stats[0], 2)
                                    dots_row['SleepTime'] = round(stats[1], 2)
                                    dots_row['Jitter'] = round(stats[2], 2)
                                    dots_row['Process_Path_Count'] = stats[3]
                                    dots_row['SVM_Probability'] = str(round(svm_proba * 100, 2)) + "%"
                                    dots_row['LogReg_Probability'] = str(round(logreg_proba * 100, 2)) + "%"
                                    dots_row['RF_Probability'] = str(round(rf_proba * 100, 2)) + "%"
                                    dots_row['Voting_Probability'] = str(round(voting_proba * 100, 2)) + "%"


                                    payload_string = "Event=\"Machine Learning Detection Algorithm for Cyber Security\", Payload=" + json.dumps(dots_row)
                                    payload_string = payload_string.replace("'","")
                                    payload_string = payload_string.replace(": ", ":")
                                    payload_string = payload_string.replace('\\\\', '\\')
                                    #print(payload_string)

                                    try:
                                        #send_syslog.send_syslog(Qradar_address, 514, "AIDA-ML", payload_string)
                                        run_log.run_log("AIDA-ML", payload_string)
                                        alert_count += 1
                                    except:
                                        run_log.run_log("ERROR", "Failed to send alert to Qradar: " + payload_string)

    run_log.run_log("INFO", "15. Number of alerts sent to Qradar: " + str(alert_count) + " * Number of Hostname in AQL: " + str(hostname_count))"""

    #### group by DATE
    grouped_DATE = other_functions.groupby(data, 'Date')
    for each_date, events_by_date in grouped_DATE.items():

        #### group by Time_interval
        grouped_by_interval = other_functions.groupby(events_by_date, 'Time_interval')
        for each_interval, events_by_interval in grouped_by_interval.items():

            #### group by HOSTNAME
            grouped_by_hostname = other_functions.groupby(events_by_interval, 'sysmon_hostname (custom)')  # Check actual field name
            for each_hostname, events_by_hostname in grouped_by_hostname.items():
                hostname_count += 1

                #### group by path
                grouped_by_path = other_functions.groupby(events_by_hostname, 'Process Path (custom)')  # Check actual field name
                for each_path, events_in_group in grouped_by_path.items():

                    #### Calculate dots grouped by path
                    if each_date == get_dot_DATE:

                        timestamps = [event['Time'] for event in events_in_group]
                        
                        # The last event in the group can be used to get common properties
                        last_event_in_group = events_in_group[-1]

                        # Construct the correct key for the counter
                        current_key = (
                            each_date,
                            each_interval,
                            each_hostname,
                            each_path
                        )
                        process_path_occurence_count = process_path_counts.get(current_key, 0)
                        #print("current key:", current_key)
                        #print("process_path_occurence_count:", process_path_occurence_count)

                        if len(timestamps)> fetch_data_frequency:
                            stats = calculate_stats.calculate_stats(timestamps, process_path_occurence_count)

                            if stats[0]>1 and stats[1]>0 and stats[2] != False and stats[3] >1:
                                # Create features list once for all models
                                features = [[stats[0], stats[1], stats[2], stats[3]]]
                                
                                # Predict with all three models
                                svm_result = svm_model_hostname.predict(features)
                                svm_proba = svm_model_hostname.predict_proba(features)[0][1]
                                logreg_result = logreg_model_hostname.predict(features)
                                logreg_proba = logreg_model_hostname.predict_proba(features)[0][1]
                                rf_result = rf_model_hostname.predict(features)
                                rf_proba = rf_model_hostname.predict_proba(features)[0][1]

                                # Voting model prediction
                                voting_result = voting_model_hostname.predict(features)
                                voting_proba = voting_model_hostname.predict_proba(features)[0][1]
                                
                                # Alert if any model detects attack (remove 0.8 probability threshold)
                                if (svm_result[0] == 1) or (logreg_result[0] == 1) or (rf_result[0] == 1) or (voting_result[0] == 1):
                                    dots_row = {}
                                    dots_row['Alert_ID'] = 2
                                    dots_row['Alert_Description'] = "Suspect C2 Traffic"
                                    dots_row['Confident_Level'] = str(float(round(np.mean([svm_proba, logreg_proba, rf_proba, voting_proba]) * 100, 2))) + "%"
                                    dots_row['Date'] = last_event_in_group['Date']
                                    dots_row['Time'] = get_dot_STARTTIME + "-" + get_dot_ENDTIME
                                    dots_row['System_Hostname'] = each_hostname
                                    dots_row['Source IP'] = last_event_in_group['Source IP']
                                    # Fix string escape for backslash
                                    dots_row['Process_Path'] = str(each_path).replace('\\\\', '\\')
                                    dots_row['Runtime'] = round(stats[0], 2)
                                    dots_row['SleepTime'] = round(stats[1], 2)
                                    dots_row['Jitter'] = round(stats[2], 2)
                                    dots_row['Process_Path_Count'] = stats[3]
                                    dots_row['SVM_Probability'] = str(round(svm_proba * 100, 2)) + "%"
                                    dots_row['LogReg_Probability'] = str(round(logreg_proba * 100, 2)) + "%"
                                    dots_row['RF_Probability'] = str(round(rf_proba * 100, 2)) + "%"
                                    dots_row['Voting_Probability'] = str(round(voting_proba * 100, 2)) + "%"


                                    payload_string = "Event=\"Machine Learning Detection Algorithm for Cyber Security\", Payload=" + json.dumps(dots_row)
                                    payload_string = payload_string.replace("'","")
                                    payload_string = payload_string.replace(": ", ":")
                                    payload_string = payload_string.replace('\\\\', '\\')
                                    #print(payload_string)

                                    try:
                                        #send_syslog.send_syslog(Qradar_address, 514, "AIDA-ML", payload_string)
                                        run_log.run_log("AIDA-ML", payload_string)
                                        alert_count += 1
                                    except:
                                        run_log.run_log("ERROR", "Failed to send alert to Qradar: " + payload_string)

    run_log.run_log("INFO", "15. Number of alerts sent to Qradar: " + str(alert_count) + " * Number of Hostname in AQL: " + str(hostname_count))
    
    run_log.run_log("INFO", "16. The program is finished\n")
    

if __name__ == "__main__":
    main()