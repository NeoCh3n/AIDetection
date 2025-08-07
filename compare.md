
+--------------------------+      +--------------------------+
| Training_data/           |      | AQL JSON Query           |
| Normal Activity (CSV)    |      | (via API/mongodb_ingester) |
| Picus Attack Data (CSV)  |      |                          |
+------------+-------------+      +-------------+------------+
             |                           |
             ▼                           ▼
+-------------------------------------------------------------+
|                     data_loader.py                           |  <--  (被 pipeline_controller.py 调用)
|  • Mode: 'train' -> Reads CSV                                |
|  • Mode: 'detect' -> Reads JSON from MongoDB                 |
|  • **Inline Preprocessing:**                                 |
|    - Parses timestamps using time_utils.py                   |
|    - Ensures correct data types (int, string)                |
|  • **Output: Clean, Standardized Pandas DataFrame**          |
|[hostname, rule_id, timestamp, count, optional: source_label] |
+----------------------------+----------------------------------+
                             |
                             ▼
+-------------------------------------------------------------+
|                   feature_aggregator.py                     |
|  • Groups data by a unique window_id and hostname          |
|  • Aggregates Rule counts within each 30-minute window     |
|  • **Output: Aggregated DataFrame**                        |
| [window_id, hostname, aggregated_rules_dict, optional: label] |
+----------------------------+----------------------------------+
                             |
                             ▼
+-------------------------------------------------------------+
|                    feature_generator.py                     |
|  • Maps aggregated_rules_dict to a 1500-dim sparse vector  |  <-- (共享 rule_manager.py)
|   (The 1500 here is just an assumption, the real number of |
|   rules needs to be extracted from the Qradar_rule folder) |
|  • Converts sparse vector to dense for the model           |
|  • Adds final label column (`is_attack`) if in 'train' mode |
|  • **Output: Final Feature Matrix (X) and Labels (y)**     |
+----------------------------+----------------------------------+
                             |
         +-------------------+------------------+
         | (Mode: 'train')                   | (Mode: 'detect')
         ▼                                   ▼
+--------------------------+      +--------------------------+
| model_training.py        |      | model_predictor.py       |
|  • Splits data           |      |  • Loads model           |
|  • Trains RF model       |      |  • Makes prediction      |
|  • Saves model (.joblib) |      |  • Returns result        |
+--------------------------+      +--------------------------+
         |                                   |
         ▼                                   ▼
+--------------------------+      +--------------------------+
| model_evaluation.py      |      | Output: Prediction Result|
|  • Loads test data & model |    +--------------------------+
|  • Calculates metrics    |
|  • Feature importances   |
+--------------------------+


================================================================================================================
|                                         PART 1: DATA ACQUISITION LAYER                                         |
================================================================================================================
                                                                                                                   
  TRAINING DATA SOURCE (Offline, Manual)             DETECTION DATA SOURCE (Online, Scheduled)
  **************************************             *******************************************
                                                                                                                   
+------------------------------------+             START (Scheduled Job, e.g., every 30 mins)
| Training_data/                     |                          |
|  - normal_activity.csv             |                          V
|  - picus_attack_ransomware.csv     |             +--------------------------------+
+------------------------------------+             |   create_searches_Qradar.py    |
                                                   |   (Initiate AQL Search Job)    |
                                                   +--------------------------------+
                                                                 |
                                                                 V
                                                   +--------------------------------+
                                                   |   status_searches_Qradar.py    |
                                                   |   (Poll Job Status)            |
                                                   +--------------------------------+
                                                                 |
                                                                 V
                                                   +--------------------------------+
                                                   |   result_searches_Qradar.py    |
                                                   |   (Fetch Results as JSON)      |
                                                   +--------------------------------+
                                                                 |
                                                                 V
                                                   +--------------------------------+
                                                   |   insert_DB.py (mongodb)       |
                                                   |   (Stage raw events in MongoDB)|
                                                   +--------------------------------+
                                                                 |
                                                                 V
                                                   +--------------------------------+
                                                   |   delete_searches_Qradar.py    |
                                                   |   (Cleanup QRadar Search Job)  |
                                                   +--------------------------------+
                                                                 |
                                                  (Raw JSON events now in MongoDB)


================================================================================================================
|                                       PART 2: UNIFIED PROCESSING PIPELINE                                      |
================================================================================================================
                                                        
 (Invoked by pipeline_controller.py with mode='train' or 'detect')
                                                        
                          +-------------------------------------------------------------+
                          |                     data_loader.py                            |
                          |  • Mode 'train': Reads from Training_data/ CSV files        |
                          |  • Mode 'detect': Queries recent data from MongoDB          |
                          |                                                             |
                          +-------------------------------------------------------------+
                                                        |
                                                        V
                          +-------------------------------------------------------------+
                          |                   feature_aggregator.py                     |
                          |  • Groups by hostname & 30-min window                     |
                          |  • Aggregates Rule counts                                   |
                          +-------------------------------------------------------------+
                                                        |
                                                        V
                          +-------------------------------------------------------------+
                          |                    feature_generator.py                     |
                          |  • Vectorizes aggregated data into 1500-dim vectors       |
                          |  • Adds `is_attack` label (in 'train' mode only)            |
                          +-------------------------------------------------------------+
                                                        |
                               (Final Feature Matrix X (and y for training) is ready)


================================================================================================================
|                                        PART 3: MODEL ACTION LAYER                                              |
================================================================================================================
                                                        
 (Invoked by pipeline_controller.py based on mode)
                                                        
+----------------------------------+                               +----------------------------------+
| IF mode == 'train':              |                               | IF mode == 'detect':             |
|                                  |                               |                                  |
|   +--------------------------+   |                               |   +--------------------------+   |
|   |   model_training.py      |   |                               |   |   model_predictor.py     |   |
|   |                          |   |                               |   |                          |   |
|   +--------------------------+   |                               |   +--------------------------+   |
|              |                   |                               |              |                   |
|              V                   |                               |              V                   |
|   +--------------------------+   |                               |   (Malicious?) --------------YES-+
|   |   model_evaluation.py    |   |                               |       | NO                       |
|   |                          |   |                               |       V                          |
|   +--------------------------+   |                               |      END                         |
|              |                   |                               |                                  |
|              V                   |                               |   +--------------------------+   |
|             END                  |                               |   |      run_log.py          |   |
|                                  |                               |   |      (Log detection)     |   |
+----------------------------------+                               |   +--------------------------+   |
                                                                   |              |                   |
                                                                   |              V                   |
                                                                   |   +--------------------------+   |
                                                                   |   |    send_syslog.py        |   |
                                                                   |   |    (Alert QRadar)        |   |
                                                                   |   +--------------------------+   |
                                                                   |              |                   |
                                                                   |              V                   |
                                                                   |             END                  | 
                                                                   |                                  |
                                                                    +----------------------------------+
