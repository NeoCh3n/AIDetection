# File: train_ML_model.py

import os
import sys
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
import csv
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from collections import Counter
import time
import math
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# from calculate_stats import calculate_stats # Assuming calculate_stats is defined elsewhere or needed later

#### change parameters here
time_interval = 30

normal_logs_folder_path = os.path.join("Training_Data", "normal_logs_aida")
attack_logs_folder_path = os.path.join("Training_Data", "attack_logs_aida")

svm_model = make_pipeline(StandardScaler(), SVC(gamma='scale', kernel='poly', degree=3, decision_function_shape='ovr', probability=True, cache_size = 10000000, class_weight = 'balanced'))
logreg_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs'))  # solver默认支持概率
rf_model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))  # RF默认支持概率
# LogisticRegression 和 RandomForestClassifier 都自带概率输出，无需额外参数，只需在预测时用 predict_proba 即可。

#### DO NOT CHANGE THE BELOW
#### DO NOT CHANGE THE BELOW


####################
#### read CSV data
normal_data = []
attack_data = []

#### read normal data from folder
print("Normal data:")
for normal_csv in os.listdir(normal_logs_folder_path):
    print(normal_csv)
    with open(os.path.join(normal_logs_folder_path, normal_csv)) as each_csvfile_DATE:

        csvdata = []
        reader = csv.reader(each_csvfile_DATE)
        header = []
        row_num = 0
        for eachrow in reader:
            row_num += 1
            if len(header) == 0:
                header = eachrow

            else:
                eachrow_dict = {}
                # map LOG data key
                log_source_time = eachrow[header.index('Log Source Time')]
                eachrow_time = datetime.datetime.strptime(log_source_time, "%b %d, %Y, %I:%M:%S %p")
                eachrow_dict['Date'] = eachrow_time.strftime("%Y-%m-%d")
                eachrow_dict['Time'] = eachrow_time.strftime("%H:%M:%S")

                eachrow_time_minute = str(eachrow_time.strftime("%M"))
                if eachrow_time_minute[0] == "0" or eachrow_time_minute[0] == "1" or eachrow_time_minute[0] == "2":
                    interval_starttime = str(eachrow_time.strftime("%H")) + ":" + "00"
                    interval_endtime = str(eachrow_time.strftime("%H")) + ":" + "29"
                else:
                    interval_starttime = str(eachrow_time.strftime("%H")) + ":" + "30"
                    interval_endtime = str(eachrow_time.strftime("%H")) + ":" + "59"
                eachrow_dict['Time_interval'] = interval_starttime + "-" + interval_endtime

                for i in range(len(eachrow)):
                    eachrow_dict[header[i]] = eachrow[i]
                normal_data.append(eachrow_dict)
        print(row_num)


#### read attack data from folder
print("Attack data:")
for attack_csv in os.listdir(attack_logs_folder_path):
    print(attack_csv)
    with open(os.path.join(attack_logs_folder_path, attack_csv)) as each_csvfile_DATE:

        csvdata = []
        reader = csv.reader(each_csvfile_DATE)
        header = []
        row_num = 1 # Changed from 0 to 1 based on logic below
        for eachrow in reader:
            row_num += 1
            if len(header) == 0:
                 header = eachrow

            else:
                eachrow_dict = {}
                #### ADD DATE key
                log_source_time = eachrow[header.index('Log Source Time')]
                eachrow_time = datetime.datetime.strptime(log_source_time, "%b %d, %Y, %I:%M:%S %p")
                eachrow_dict['Date'] = eachrow_time.strftime("%Y-%m-%d")
                eachrow_dict['Time'] = eachrow_time.strftime("%H:%M:%S")

                eachrow_time_minute = str(eachrow_time.strftime("%M"))
                if eachrow_time_minute[0] == "0" or eachrow_time_minute[0] == "1" or eachrow_time_minute[0] == "2":
                    interval_starttime = str(eachrow_time.strftime("%H")) + ":" + "00"
                    interval_endtime = str(eachrow_time.strftime("%H")) + ":" + "29"
                else:
                    interval_starttime = str(eachrow_time.strftime("%H")) + ":" + "30"
                    interval_endtime = str(eachrow_time.strftime("%H")) + ":" + "59"
                eachrow_dict['Time_interval'] = interval_starttime + "-" + interval_endtime

                for i in range(len(eachrow)):
                    eachrow_dict[header[i]] = eachrow[i]
                attack_data.append(eachrow_dict)
        print(row_num)


#### Group the data
#####################################
#### devide csv into groups

def groupby(csvdata, column_name):
    groupeddata = {}
    for eachrow in csvdata:
        if eachrow[column_name] not in groupeddata:
            groupeddata[eachrow[column_name]] = [eachrow]

        else:
            groupeddata[eachrow[column_name]].append(eachrow)

    return (groupeddata)

#####################################
##### get normal dots
normal_date_timeint_hostname_DOTS = []

normal_date_timeint_hostname_path_DOTS = []



all_dots_row = []

# Count occurrences of each process path in normal data
"""normal_process_path_counts = Counter()
for row in normal_data:
    if 'Process Path (custom)' in row:
        normal_process_path_counts[row['Process Path (custom)']] += 1"""
normal_process_path_counts = Counter((row['Date'],row['Time_interval'],row['sysmon_hostname (custom)'],row['Process Path (custom)']) for row in normal_data if 'Process Path (custom)' in row and 'sysmon_hostname (custom)' in row)

#####################################
# group by DATE
normal_grouped_DATE = groupby(normal_data,'Date')
for each_date in normal_grouped_DATE:

    # group by TIME INTERVAL
    grouped_DATE_TIMEINT = groupby(normal_grouped_DATE[each_date],'Time_interval')
    for each_date_timeint in grouped_DATE_TIMEINT:

        # group by sysmon_hostname
        grouped_DATE_TIMEINT_HOSTNAME = groupby(grouped_DATE_TIMEINT[each_date_timeint],'sysmon_hostname (custom)')
        for each_date_timeint_hostname in grouped_DATE_TIMEINT_HOSTNAME:

            # group by PATH
            grouped_DATE_TIMEINT_HOSTNAME_PATH = groupby(grouped_DATE_TIMEINT_HOSTNAME[each_date_timeint_hostname],'Process Path (custom)')

            for each_date_timeint_hostname_path in grouped_DATE_TIMEINT_HOSTNAME_PATH:
                timestamps = []
                for time_str in grouped_DATE_TIMEINT_HOSTNAME_PATH[each_date_timeint_hostname_path]:
                     timestamps.append(time_str['Time'])

                if len(timestamps) >4:
                     # Get the count of this process path
                     #process_path_count = normal_process_path_counts[each_date_timeint_hostname_path]
                     process_path_count = normal_process_path_counts.get((each_date,each_date_timeint,each_date_timeint_hostname,each_date_timeint_hostname_path),0)
                     if len(timestamps) > 1:
                         ts_objects = [datetime.datetime.strptime(ts, "%H:%M:%S") for ts in timestamps]
                         ts_seconds = [(t - datetime.datetime.combine(datetime.date(1900, 1, 1), datetime.time.min)).total_seconds() for t in ts_objects]
                         intervals = np.diff(sorted(ts_seconds))
                         avg_interval = np.mean(intervals) if len(intervals) > 0 else 0
                         std_dev_interval = np.std(intervals) if len(intervals) > 0 else 0
                         cv_interval = (std_dev_interval / avg_interval) * 100 if avg_interval != 0 else 0
                         stat = [avg_interval, std_dev_interval, cv_interval, process_path_count] # [AVERAGE_DOT, SD_DOT, CV_DOT, PATH_COUNT]
                     else:
                         stat = [0, 0, 0, process_path_count] # Default if not enough data points

                     dots_row = {}
                     dots_row['Date'] = each_date
                     dots_row['Time_interval'] = each_date_timeint
                     dots_row['sysmon_hostname (custom)'] = each_date_timeint_hostname
                     dots_row['Process Path (custom)'] = each_date_timeint_hostname_path

                     dots_row['AVERAGE_DOT'] = stat[0]
                     dots_row['SD_DOT'] = stat[1]
                     dots_row['CV_DOT'] = stat[2]
                     dots_row['PATH_COUNT'] = stat[3]  # Add the process path count
                     # 放宽筛选条件，全部加入
                     normal_date_timeint_hostname_path_DOTS.append(dots_row)
                     all_dots_row.append(dots_row)

print("Normal path groups:", len(normal_date_timeint_hostname_path_DOTS))  # 在for循环后加

#####################################
##### get attack dots
attack_date_timeint_hostname_DOTS = []

attack_date_timeint_hostname_path_DOTS = []


# Count occurrences of each process path in attack data
"""attack_process_path_counts = Counter()
for row in attack_data:
    if 'Process Path (custom)' in row:
        attack_process_path_counts[row['Process Path (custom)']] += 1"""
attack_process_path_counts = Counter((row['Date'],row['Time_interval'],row['sysmon_hostname (custom)'],row['Process Path (custom)'])for row in attack_data if 'Process Path (custom)' in row and 'sysmon_hostname (custom)' in row)

# group by DATE
attack_grouped_DATE = groupby(attack_data,'Date')
for each_date in attack_grouped_DATE:

    # group by TIME INTERVAL
    grouped_DATE_TIMEINT = groupby(attack_grouped_DATE[each_date],'Time_interval')
    for each_date_timeint in grouped_DATE_TIMEINT:
        if True:
            # group by HOSTNAME
            grouped_DATE_TIMEINT_HOSTNAME = groupby(grouped_DATE_TIMEINT[each_date_timeint],'sysmon_hostname (custom)')
            for each_date_timeint_hostname in grouped_DATE_TIMEINT_HOSTNAME:

                # group by PATH
                grouped_DATE_TIMEINT_HOSTNAME_PATH = groupby(grouped_DATE_TIMEINT_HOSTNAME[each_date_timeint_hostname],'Process Path (custom)')
                for each_date_timeint_hostname_path in grouped_DATE_TIMEINT_HOSTNAME_PATH:
                    if True:
                        timestamps = []
                        for time_str in grouped_DATE_TIMEINT_HOSTNAME_PATH[each_date_timeint_hostname_path]:
                             timestamps.append(time_str['Time'])

                        if len(timestamps) >4:
                            # Get the count of this process path
                            #process_path_count = attack_process_path_counts[each_date_timeint_hostname_path]
                            process_path_count = attack_process_path_counts.get((each_date,each_date_timeint,each_date_timeint_hostname,each_date_timeint_hostname_path),0)
                            if len(timestamps) > 1:
                                ts_objects = [datetime.datetime.strptime(ts, "%H:%M:%S") for ts in timestamps]
                                ts_seconds = [(t - datetime.datetime.combine(datetime.date(1900, 1, 1), datetime.time.min)).total_seconds() for t in ts_objects]
                                intervals = np.diff(sorted(ts_seconds))
                                avg_interval = np.mean(intervals) if len(intervals) > 0 else 0
                                std_dev_interval = np.std(intervals) if len(intervals) > 0 else 0
                                cv_interval = (std_dev_interval / avg_interval) * 100 if avg_interval != 0 else 0
                                stat = [avg_interval, std_dev_interval, cv_interval, process_path_count] # [AVERAGE_DOT, SD_DOT, CV_DOT, PATH_COUNT]
                            else:
                                stat = [0, 0, 0, process_path_count] # Default if not enough data points

                            dots_row = {}
                            dots_row['Date'] = each_date
                            dots_row['Time_interval'] = each_date_timeint
                            dots_row['sysmon_hostname (custom)'] = each_date_timeint_hostname
                            dots_row['Process Path (custom)'] = each_date_timeint_hostname_path

                            dots_row['AVERAGE_DOT'] = stat[0]
                            dots_row['SD_DOT'] = stat[1]
                            dots_row['CV_DOT'] = stat[2]
                            dots_row['PATH_COUNT'] = stat[3]  # Add the process path count
                            # 放宽筛选条件，全部加入
                            attack_date_timeint_hostname_path_DOTS.append(dots_row)
                            all_dots_row.append(dots_row)

print("Attack path groups:", len(attack_date_timeint_hostname_path_DOTS))

normal_date_timeint_hostname_path_AVERAGE_DOTS = []
for each in normal_date_timeint_hostname_path_DOTS:
    normal_date_timeint_hostname_path_AVERAGE_DOTS.append([each['AVERAGE_DOT']])

normal_date_timeint_hostname_path_SD_DOTS = []
for each in normal_date_timeint_hostname_path_DOTS:
    normal_date_timeint_hostname_path_SD_DOTS.append([each['SD_DOT']])

normal_date_timeint_hostname_path_CV_DOTS = []
for each in normal_date_timeint_hostname_path_DOTS:
    normal_date_timeint_hostname_path_CV_DOTS.append([each['CV_DOT']])

normal_date_timeint_hostname_path_PATH_COUNTS = []
for each in normal_date_timeint_hostname_path_DOTS:
    normal_date_timeint_hostname_path_PATH_COUNTS.append([each['PATH_COUNT']])


attack_date_timeint_hostname_path_AVERAGE_DOTS = []
for each in attack_date_timeint_hostname_path_DOTS:
    attack_date_timeint_hostname_path_AVERAGE_DOTS.append([each['AVERAGE_DOT']])

attack_date_timeint_hostname_path_SD_DOTS = []
for each in attack_date_timeint_hostname_path_DOTS:
    attack_date_timeint_hostname_path_SD_DOTS.append([each['SD_DOT']])

attack_date_timeint_hostname_path_CV_DOTS = []
for each in attack_date_timeint_hostname_path_DOTS:
    attack_date_timeint_hostname_path_CV_DOTS.append([each['CV_DOT']])

attack_date_timeint_hostname_path_PATH_COUNTS = []
for each in attack_date_timeint_hostname_path_DOTS:
    attack_date_timeint_hostname_path_PATH_COUNTS.append([each['PATH_COUNT']])


#####################################
#### Train 3 model
#### data grouped by hostname

#### Make the list to 4D (adding process path count)
train_data_hostname_4d = []

normal_count = 0
#### select dots for training，Jitter（CV）只保留0-99
for i in range(len(normal_date_timeint_hostname_path_AVERAGE_DOTS)):
    cv = normal_date_timeint_hostname_path_CV_DOTS[i][0]
    if normal_date_timeint_hostname_path_AVERAGE_DOTS[i] and 0 <= cv < 100:
        train_data_hostname_4d.append([
            normal_date_timeint_hostname_path_AVERAGE_DOTS[i][0],
            normal_date_timeint_hostname_path_SD_DOTS[i][0],
            cv,
            normal_date_timeint_hostname_path_PATH_COUNTS[i][0]
        ])
        normal_count+=1

#### add attack dots to train_data_4d，Jitter（CV）只保留0-99
for i in range(len(attack_date_timeint_hostname_path_AVERAGE_DOTS)):
    cv = attack_date_timeint_hostname_path_CV_DOTS[i][0]
    if attack_date_timeint_hostname_path_AVERAGE_DOTS[i] and 0 <= cv < 100:
        train_data_hostname_4d.append([
            attack_date_timeint_hostname_path_AVERAGE_DOTS[i][0],
            attack_date_timeint_hostname_path_SD_DOTS[i][0],
            cv,
            attack_date_timeint_hostname_path_PATH_COUNTS[i][0]
        ])


#### get training label
train_label= []
for i in range(len(train_data_hostname_4d)):
    if i < normal_count:
        train_label.append(0)
    else:
        train_label.append(1)

print("train_label Counter:", Counter(train_label))
####################
#### Train SVM model
print("Training SVM...")
svm_model.fit(train_data_hostname_4d, train_label)
print("Training Logistic Regression...")
logreg_model.fit(train_data_hostname_4d, train_label)
print("Training Random Forest...")
rf_model.fit(train_data_hostname_4d, train_label)

######################
#### Train Voting model
print("Training Voting model...")
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('svm',svm_model),
        ('logreg',logreg_model),
        ('rf',rf_model)
    ],
    voting='soft' #'hard' fr majority viting, 'soft' for averagig probabilities
)
voting_clf.fit(train_data_hostname_4d,train_label)

#### Saving models
print("Saving SVM model...")
joblib.dump(svm_model, os.path.join("Model", "svm_NetworkConnection_4D"))
print("Saving Logistic Regression model...")
joblib.dump(logreg_model, os.path.join("Model", "logreg_NetworkConnection_4D"))
print("Saving Random Forest model...")
joblib.dump(rf_model, os.path.join("Model", "rf_NetworkConnection_4D"))

print("Saving Voting Classifier model...")
joblib.dump(voting_clf,os.path.join("Model", "voting_NetworkConnection_4D"))

#### Loading models
print("Loading SVM model...")
svm_model_clf = joblib.load(os.path.join("Model", "svm_NetworkConnection_4D"))
print("Loading Logistic Regression model...")
logreg_model_clf = joblib.load(os.path.join("Model", "logreg_NetworkConnection_4D"))
print("Loading Random Forest model...")
rf_model_clf = joblib.load(os.path.join("Model", "rf_NetworkConnection_4D"))


####################
#### Predict SVM
predict_data_4d = []
#### add normal dots to predict data 4d，Jitter（CV）只保留0-99
for i in range(len(normal_date_timeint_hostname_path_AVERAGE_DOTS)):
    cv = normal_date_timeint_hostname_path_CV_DOTS[i][0]
    if 0 <= cv < 100:
        predict_data_4d.append([
            normal_date_timeint_hostname_path_AVERAGE_DOTS[i][0],
            normal_date_timeint_hostname_path_SD_DOTS[i][0],
            cv,
            normal_date_timeint_hostname_path_PATH_COUNTS[i][0]
        ])

#### add attack dots to predict data 4d，Jitter（CV）只保留0-99
for i in range(len(attack_date_timeint_hostname_path_AVERAGE_DOTS)):
    cv = attack_date_timeint_hostname_path_CV_DOTS[i][0]
    if 0 <= cv < 100:
        predict_data_4d.append([
            attack_date_timeint_hostname_path_AVERAGE_DOTS[i][0],
            attack_date_timeint_hostname_path_SD_DOTS[i][0],
            cv,
            attack_date_timeint_hostname_path_PATH_COUNTS[i][0]
        ])


print("Predicting...")
predict_result = svm_model_clf.predict(predict_data_4d)
predict_result_logreg = logreg_model_clf.predict(predict_data_4d)
predict_result_rf = rf_model_clf.predict(predict_data_4d)

# 写入预测结果到 result.csv
with open("result.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["avg", "sd", "cv", "path_count", "label"])  # 写表头
    for i in range(len(predict_result)):
        row = predict_data_4d[i] + [predict_result[i]]
        writer.writerow(row)
with open("result_logreg.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["avg", "sd", "cv", "path_count", "label"])
    for i in range(len(predict_result_logreg)):
        row = predict_data_4d[i] + [predict_result_logreg[i]]
        writer.writerow(row)
with open("result_rf.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["avg", "sd", "cv", "path_count", "label"])
    for i in range(len(predict_result_rf)):
        row = predict_data_4d[i] + [predict_result_rf[i]]
        writer.writerow(row)

# Optionally print attack counts for each model
attacknum_logreg = sum(1 for x in predict_result_logreg if x == 1)
attacknum_rf = sum(1 for x in predict_result_rf if x == 1)
print("Logistic Regression red dots", attacknum_logreg)
print("Random Forest red dots", attacknum_rf)

predicted_blue_dots = []
predicted_red_dots = []
attacknum = 0
for i in range(len(predict_result)):
    if predict_result[i] == 0:
        predicted_blue_dots.append(predict_data_4d[i])
    else:
        predicted_red_dots.append(predict_data_4d[i])
        attacknum +=1

print("Total dots",len(predict_result))
print("red dots", attacknum)


#####################################
#### plot the graph before prediction - now with 4D visualization
fig = plt.figure(figsize=(15, 10))

# First subplot - original 3D visualization
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.set_xlabel("run time (min)")
ax1.set_ylabel("sleep time (s)")
ax1.set_zlabel("jitter (%)")

# Extracting features correctly for plotting
normal_avg = [dot[0]/60 for dot in train_data_hostname_4d[:normal_count]]
normal_sd = [dot[1] for dot in train_data_hostname_4d[:normal_count]]
normal_cv = [dot[2] for dot in train_data_hostname_4d[:normal_count]]
normal_pc = [dot[3] for dot in train_data_hostname_4d[:normal_count]]

attack_avg = [dot[0]/60 for dot in train_data_hostname_4d[normal_count:]]
attack_sd = [dot[1] for dot in train_data_hostname_4d[normal_count:]]
attack_cv = [dot[2] for dot in train_data_hostname_4d[normal_count:]]
attack_pc = [dot[3] for dot in train_data_hostname_4d[normal_count:]]

ax1.scatter(normal_avg, normal_sd, normal_cv, color='blue', marker='o', alpha=0.1)
ax1.scatter(attack_avg, attack_sd, attack_cv, color='red', marker='o', alpha=0.25)
ax1.set_title("3D view: Runtime (min), Sleep Time (s), Jitter (%)")

# Second subplot - showing path count dimension
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.set_xlabel("run time (min)")
ax2.set_ylabel("sleep time (s)")
ax2.set_zlabel("process path count")

ax2.scatter(normal_avg, normal_sd, normal_pc, color='blue', marker='o', alpha=0.1)
ax2.scatter(attack_avg, attack_sd, attack_pc, color='red', marker='o', alpha=0.25)
ax2.set_title("3D view: Runtime (min), Sleep Time (s), Process Path Count")

# Third subplot - another view with path count
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.set_xlabel("run time")
ax3.set_ylabel("jitter")
ax3.set_zlabel("process path count")

ax3.scatter(normal_avg, normal_cv, normal_pc, color='blue', marker='o', alpha=0.1)
ax3.scatter(attack_avg, attack_cv, attack_pc, color='red', marker='o', alpha=0.25)
ax3.set_title("3D view: Runtime, Jitter, Process Path Count")

# Save the 4D visualization
plt.tight_layout()
plt.savefig("plot_4d_features.png", format='png')

# Original viewpoints for compatibility
fig_3d = plt.figure(figsize=(10, 5))
ax = fig_3d.add_subplot(projection='3d')
ax.set_xlabel("run time")
ax.set_ylabel("sleep time")
ax.set_zlabel("jitter")
ax.scatter(normal_avg, normal_sd, normal_cv, color='blue', marker='o', alpha=0.1)
ax.scatter(attack_avg, attack_sd, attack_cv, color='red', marker='o', alpha=0.25)

azim=90
ax.view_init(elev=0, azim=0)
plt.savefig("plot_0_0.png", format='png')

ax.view_init(elev=90, azim=0)
plt.savefig("plot_90_0.png", format='png')

ax.view_init(elev=0, azim=90)
plt.savefig("plot_0_90.png", format='png')


#####################################
#### plot the prediction result with 4D visualization
fig_pred = plt.figure(figsize=(15, 10))

# First subplot - prediction with original 3D
ax1 = fig_pred.add_subplot(2, 2, 1, projection='3d')
ax1.set_xlabel("run time")
ax1.set_ylabel("sleep time")
ax1.set_zlabel("jitter")

# Extract coordinates for predicted dots
blue_avg = [dot[0]/60 for dot in predicted_blue_dots]
blue_sd = [dot[1] for dot in predicted_blue_dots]
blue_cv = [dot[2] for dot in predicted_blue_dots]
blue_pc = [dot[3] for dot in predicted_blue_dots]

red_avg = [dot[0]/60 for dot in predicted_red_dots]
red_sd = [dot[1] for dot in predicted_red_dots]
red_cv = [dot[2] for dot in predicted_red_dots]
red_pc = [dot[3] for dot in predicted_red_dots]

if predicted_blue_dots:
    ax1.scatter(blue_avg, blue_sd, blue_cv, color='blue', marker='o', alpha=0.1)
if predicted_red_dots:
    ax1.scatter(red_avg, red_sd, red_cv, color='red', marker='o', alpha=0.25)
ax1.set_title("Prediction: Runtime (min), Sleep Time (s), Jitter (s)")

# Second subplot - prediction with path count
ax2 = fig_pred.add_subplot(2, 2, 2, projection='3d')
ax2.set_xlabel("run time (min)")
ax2.set_ylabel("sleep time (s)")
ax2.set_zlabel("process path count")

if predicted_blue_dots:
    ax2.scatter(blue_avg, blue_sd, blue_pc, color='blue', marker='o', alpha=0.1)
if predicted_red_dots:
    ax2.scatter(red_avg, red_sd, red_pc, color='red', marker='o', alpha=0.25)
ax2.set_title("Prediction: Runtime, Sleep Time, Process Path Count")

# Third subplot - another prediction view with path count
ax3 = fig_pred.add_subplot(2, 2, 3, projection='3d')
ax3.set_xlabel("run time (min)")
ax3.set_ylabel("jitter (s)")
ax3.set_zlabel("process path count")

if predicted_blue_dots:
    ax3.scatter(blue_avg, blue_cv, blue_pc, color='blue', marker='o', alpha=0.1)
if predicted_red_dots:
    ax3.scatter(red_avg, red_cv, red_pc, color='red', marker='o', alpha=0.25)
ax3.set_title("Prediction: Runtime, Jitter, Process Path Count")

# Save the 4D prediction visualization
plt.tight_layout()
plt.savefig("prediction_4d_features.png", format='png')

# Original viewpoints for compatibility
fig_3d = plt.figure(figsize=(10, 5))
ax = fig_3d.add_subplot(projection='3d')
ax.set_xlabel("run time (min)")
ax.set_ylabel("sleep time (s)")
ax.set_zlabel("jitter (%)")

if predicted_blue_dots:
    ax.scatter(blue_avg, blue_sd, blue_cv, color='blue', marker='o', alpha=0.1)
if predicted_red_dots:
    ax.scatter(red_avg, red_sd, red_cv, color='red', marker='o', alpha=0.25)

ax.view_init(elev=0, azim=0)
plt.savefig("prediction_0_0.png", format='png')

ax.view_init(elev=90, azim=0)
plt.savefig("prediction_90_0.png", format='png')

ax.view_init(elev=0, azim=90)
plt.savefig("prediction_0_90.png", format='png')