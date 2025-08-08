# Portable QRadar Rule Processing System

## Overview
This system has been updated to work portably across different computers. All rule files are stored in the `Qradar_rule/` folder, and the system automatically detects available rule files.

## Quick Start

### 1. Initial Setup
```bash
# Check system and create basic structure
python setup_system.py
```

### 2. Download Rules (Configure First)
Edit `get_rule.py` and set your QRadar credentials:
- `QRADAR_HOST`: Your QRadar server URL (e.g., "https://your-qradar-server")
- `API_TOKEN`: Your API token from QRadar Admin → Authorized Services

Then run:
```bash
python get_rule.py
```

This will fetch rules from 3 endpoints:
- `/api/analytics/rules` → `Qradar_rule/qradar_rules.csv`
- `/api/analytics/ade_rules` → `Qradar_rule/qradar_aderules.csv`
- `/api/analytics/building_blocks` → `Qradar_rule/qradar_buildingblocks.csv`

### 3. Create Rule Mapping
```bash
python create_rule_mapping.py
```

This creates `rule_mapping.json` automatically from all CSV files in `Qradar_rule/`

### 4. Process Your Data
```bash
python qradar_to_matrix.py your_data.csv
```

## File Structure

```
AIDetection4All/
├── Qradar_rule/              # Rule files directory
│   ├── qradar_rules.csv      # Standard rules
│   ├── qradar_aderules.csv   # ADE rules
│   └── qradar_buildingblocks.csv  # Building blocks
├── rule_mapping.json         # Auto-generated rule mapping
├── get_rule.py              # Updated rule downloader
├── create_rule_mapping.py   # Auto-mapping creator
├── qradar_to_matrix.py      # Data transformer (updated)
└── setup_system.py          # Complete system setup
```

## Portable Usage

The system is now fully portable. Just copy the entire folder to any computer and:

1. Run `python setup_system.py` to check everything
2. Configure `get_rule.py` with your QRadar details
3. Run `python get_rule.py` to download rules
4. Run `python create_rule_mapping.py` to create mapping
5. Use as normal

## Configuration via Environment Variables

You can configure connections without editing code by setting these environment variables in your shell or CI:

- QRADAR_ADDRESS: Syslog/QRadar address (default: 192.168.153.123)
- QRADAR_API_TOKEN: API token used in API requests (mask this; do not commit)
- SYSLOG_ADDRESS: Destination syslog address for run_log/syslog (default: 192.168.153.123)
- SYSLOG_PORT: Destination syslog port (default: 514)
- SYSLOG_HEADER_BASE: Base name for syslog headers (default: AIR)
- SYSLOG_HEADER_ML: Syslog header for ML events (default: AIR-RF)
- SYSLOG_HEADER_LOG: Syslog header for log events (default: AIR-RF)

Example (zsh):

```bash
export QRADAR_ADDRESS="https://your-qradar"
export QRADAR_API_TOKEN="<token>"
export SYSLOG_ADDRESS="<qradar-syslog-ip>"
export SYSLOG_PORT=514
export SYSLOG_HEADER_BASE="AIR"
export SYSLOG_HEADER_ML="AIR-RF"
export SYSLOG_HEADER_LOG="AIR-RF"
```

Note: When running `python system/config.py` directly, the token will be masked in the output.

## Manual Rule File Import

If you have rule files from another source, just place them in the `Qradar_rule/` folder as CSV files with columns: `id`, `name`, `type`, `enabled`, `origin`

Then run:
```bash
python create_rule_mapping.py
```

## Troubleshooting

### Rule Files Not Found
- Check `Qradar_rule/` directory exists
- Ensure CSV files are present
- Run `python create_rule_mapping.py --validate`

### Wrong Rule Count
- Verify all rule CSV files in `Qradar_rule/`
- Check CSV format has correct column names
- Run `python create_rule_mapping.py` to regenerate mapping