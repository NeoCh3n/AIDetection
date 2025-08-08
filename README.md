# AIDetection4Ransomware - Ransomware Detection System

A supervised machine learning system specifically designed to detect ransomware activity using QRadar rule trigger frequencies. Built for enterprise-scale detection with support for both offline deployment and production environments.

## 🎯 Project Overview

This system leverages labeled datasets from Picus Breach and Attack Simulation (BAS) to train a Random Forest classifier that distinguishes between normal operational behavior and ransomware attack patterns using QRadar rule trigger frequencies as primary features.

**Key Features:**
- **30-minute sliding windows** with 15-minute query frequency
- **Detection-only mode** for production environments
- **Host-level breakdown** for precise threat response
- **Cross-platform support** (RHEL 7.9, macOS ARM)
- **Unified MongoDB architecture** for consistent data processing

## 🏗️ Architecture

### Detection Pipeline
```
QRadar AQL Queries → MongoDB Storage → ML Processing → Threat Detection
```

### Key Components
- **MongoDB Connection Utility** (`mongodb/mongodb_connection.py`) - Unified database operations
- **Time Utils** (`shared_utils/time_utils.py`) - Consistent timestamp processing
- **Data Processing** (`mongodb/insert_DB.py`) - QRadar data ingestion
- **ML Model** - Random Forest classifier for ransomware detection

## 📊 Data Schema

### Detection Windows (30-minute periods)
```json
{
  "_id": "2025-08-08_10-00-00_W0",
  "window_start": "2025-08-08T10:00:00Z",
  "window_end": "2025-08-08T10:30:00Z",
  "feature_vector": {"rule_id": count, ...},
  "host_triggers": {
    "hostname": {"total_triggers": int, "rules": {...}}
  }
}
```

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.6.8 (required)
python --version

# Install dependencies
pip install pandas==1.1.5 numpy==1.19.5 scikit-learn==0.24.2 \
             joblib==1.0.1 pymongo==3.11.0 matplotlib==3.3.4 pytz
```

### MongoDB Setup
```bash
# Run offline setup (supports RHEL 7.9 and macOS ARM)
cd mongodb
python setup_mongodb_offline.py

# Or use unified connection utility
python mongodb_connection.py
```

### Processing QRadar Data
```python
from mongodb.insert_DB import process_qradar_data

# Process QRadar search results
process_qradar_data(
    json_files=["path/to/qradar_results.json"],
    auto_cleanup=True,
    retention_days=7
)
```

## 🔧 Configuration

### MongoDB Settings (`mongodb/mongodb_config.json`)
```json
{
  "mongodb": {
    "db_name": "qradar_detection",
    "connection_string": "mongodb://localhost:27017/"
  },
  "pipeline": {
    "mode": "detection_only",
    "query_frequency_minutes": 15,
    "window_size": "30min",
    "timezone": "HKT"
  }
}
```

## 📁 Project Structure

```
AIDetection4Ransomware/
├── mongodb/                    # Database layer
│   ├── mongodb_connection.py   # Unified MongoDB operations
│   ├── insert_DB.py           # QRadar data processing
│   ├── setup_mongodb_offline.py # Cross-platform setup
│   └── mongodb_config.json    # Configuration
├── shared_utils/              # Shared utilities
│   ├── time_utils.py         # Timestamp processing
│   └── rule_manager.py       # Rule ID management
├── tests/                    # Test files
├── Qradar_rule/              # QRadar rule definitions
├── Training_data/            # CSV training datasets
└── system/                   # System utilities
```

## 🔍 Usage Examples

### 1. Setup Detection Environment
```bash
# Install and configure MongoDB
python mongodb/setup_mongodb_offline.py

# Verify MongoDB connection
python mongodb/mongodb_connection.py
```

### 2. Process QRadar Results
```bash
# Process AQL search results
python mongodb/insert_DB.py

# Check data summary
python -c "from mongodb.mongodb_connection import get_mongodb_manager; print(get_mongodb_manager().get_data_summary())"
```

### 3. Manual Data Processing
```python
from mongodb.mongodb_connection import get_mongodb_manager

with get_mongodb_manager() as db:
    # Get detection windows for specific time range
    windows = db.get_detection_windows(
        start_time=datetime(2025, 8, 1),
        end_time=datetime(2025, 8, 8)
    )
```

## 🧪 Testing

Run the test suite to verify functionality:
```bash
# Test complete pipeline
python tests/test_complete_pipeline.py

# Test data processing
python tests/test_pipeline_simple.py

# Test MongoDB operations
python tests/test_data_loader_detailed.py
```

## 📈 Performance

- **Detection Windows**: 30-minute sliding windows
- **Query Frequency**: 15 minutes for real-time detection
- **Data Retention**: 7 days for ML training data
- **Cross-platform**: Supports RHEL 7.9 and macOS ARM

## 🔒 Security Notes

- Uses simulated attack data from trusted Picus BAS platform
- Detection-only mode prevents training data storage in production
- Host-level tracking for precise threat response
- All data processing follows enterprise security standards

## 🐛 Troubleshooting

### Common Issues
1. **MongoDB Connection**: Ensure MongoDB is running on localhost:27017
2. **Timezone Issues**: Verify HKT timezone configuration in time_utils.py
3. **Rule Processing**: Check Qradar_rule folder for rule definitions
4. **Data Validation**: Use time_utils.py for consistent timestamp handling

### Debug Commands
```bash
# Check MongoDB status
python -c "from pymongo import MongoClient; MongoClient().admin.command('ping')"

# Validate timestamps
python -c "from shared_utils.time_utils import parse_qradar_timestamp; print(parse_qradar_timestamp('Aug 08, 2025, 10:20:59 AM'))"
```

## 🤝 Contributing

This is a defensive security tool designed for ransomware detection. Contributions should focus on improving detection accuracy and system reliability.

## 📄 License

Enterprise security tool - follow internal data security policies for deployment and usage.

---

**Built with** Python 3.6.8, MongoDB, scikit-learn, and enterprise-grade security practices.