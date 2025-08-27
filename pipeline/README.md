# Unified Threat Detection Pipeline

## Overview
Single orchestrator file (`main_pipeline.py`) that manages the entire threat detection flow from data acquisition to prediction.

## Usage

### Training Mode
```bash
python pipeline/main_pipeline.py train
```
- Loads training data from CSV files
- Aggregates 30-minute windows
- Trains Random Forest model
- Saves model to `./model/threat_detector.joblib`

### Detection Mode
```bash
python pipeline/main_pipeline.py detect
```
- Fetches QRadar data via API
- Stores in MongoDB
- Runs prediction on 30-minute windows
- Generates alerts for threat detection

## Architecture

### File Structure
```
pipeline/
├── main_pipeline.py      # Unified orchestrator
├── config.json          # Unified configuration
├── data_loader.py       # Data loading (existing)
├── feature_aggregator.py # 30-min window aggregation (existing)
├── feature_generator.py  # Feature vector creation (existing)
└── README.md            # This file
```

### Integration Points
- **Training**: CSV → Pipeline → Model
- **Detection**: QRadar → MongoDB → Pipeline → Prediction → Alert

## Configuration
Edit `config.json` to customize:
- QRadar connection settings
- MongoDB configuration
- Detection thresholds
- Retention policies

## Testing
All existing API integration and MongoDB modules are reused without modification.

## Usage Examples

### Scheduled Detection (30-minute intervals)
```bash
# Cron job example for 30-minute detection
*/30 * * * * cd /path/to/project && python pipeline/main_pipeline.py detect >> logs/detection.log 2>&1
```

### Manual Training
```bash
# Place training CSV files in Training_data/normal/ and Training_data/attack/
python pipeline/main_pipeline.py train
```

## Dependencies
- All existing project dependencies maintained
- No additional modules required
- Uses existing venv configuration