# OOP Threat Detection Pipeline

A modern, object-oriented implementation of the threat detection pipeline with support for multiple machine learning models and flexible configuration.

## Overview

This implementation provides:
- **Model Switching**: Easy switching between RandomForest, SVM, Logistic Regression, Gradient Boosting, and XGBoost
- **Consistent Interface**: Unified API for training and detection regardless of model type
- **Extensibility**: Easy to add new models and features
- **Configuration-Driven**: JSON-based configuration for all parameters
- **Robust Error Handling**: Comprehensive error handling and logging

## Architecture

### Core Components

1. **DataHandler**: Unified data loading from CSV files and MongoDB
2. **FeatureManipulator**: Feature engineering and preprocessing
3. **ModelBase**: Abstract base class for all ML models
4. **TrainingPipeline**: Orchestrates the complete training workflow
5. **DetectionPipeline**: Handles real-time threat detection
6. **ModelFactory**: Creates model instances based on configuration
7. **PipelineOrchestrator**: Main entry point for all operations

### Supported Models

| Model | Type | Scaling Required | Feature Importance | Notes |
|-------|------|------------------|-------------------|--------|
| Random Forest | Tree-based | No | Yes | Default, handles imbalanced data well |
| SVM | Kernel-based | Yes | No | Good for high-dimensional data |
| Logistic Regression | Linear | Yes | No | Fast, interpretable baseline |
| Gradient Boosting | Ensemble | No | Yes | sklearn implementation |
| XGBoost | Ensemble | No | Yes | Requires `pip install xgboost` |

## Usage

### Basic Training

```python
from pipeline.pipe import PipelineOrchestrator

# Create orchestrator
orchestrator = PipelineOrchestrator()

# Train with Random Forest (default)
results = orchestrator.train('random_forest')
print(f"Model saved to: {results['model_path']}")

# Train with SVM
results = orchestrator.train('svm')
```

### CLI Usage

```bash
# Train with Random Forest
python pipeline/pipe.py train --model-type random_forest

# Train with SVM
python pipeline/pipe.py train --model-type svm --config pipeline/config_oop.json

# Run detection
python pipeline/pipe.py detect --model-path ./model/threat_detector.joblib

# Verbose logging
python pipeline/pipe.py train --model-type xgboost --verbose
```

### Custom Configuration

Create a JSON configuration file (see `pipeline/config_oop.json` for example):

```python
# Use custom config
orchestrator = PipelineOrchestrator('my_config.json')
results = orchestrator.train('gradient_boosting')
```

### Detection Pipeline

```python
# Real-time detection
orchestrator = PipelineOrchestrator()
results = orchestrator.detect('./model/threat_detector_svm.joblib')

print(f"Threats detected: {results['threat_count']}")
for alert in results['alerts']:
    print(f"Alert: {alert['window_id']} - Confidence: {alert['confidence']}")
```

## Configuration

### Model-Specific Settings

Each model type can be configured with specific hyperparameters:

```json
{
  "model": {
    "random_forest": {
      "n_estimators": 200,
      "max_depth": null,
      "class_weight": "balanced_subsample"
    },
    "svm": {
      "C": 1.0,
      "kernel": "rbf",
      "class_weight": "balanced"
    },
    "xgboost": {
      "n_estimators": 100,
      "learning_rate": 0.1,
      "max_depth": 6
    }
  }
}
```

### Grid Search

Enable hyperparameter optimization:

```json
{
  "training": {
    "grid_search": {
      "enabled": true,
      "scoring": "roc_auc",
      "cv": 3,
      "param_grid": {
        "n_estimators": [100, 200],
        "max_depth": [null, 10, 20]
      }
    }
  }
}
```

## Examples

Run the comprehensive examples:

```bash
# Basic examples
python pipeline/oop_examples.py

# Interactive demo
python pipeline/oop_examples.py --interactive
```

## Model Comparison

Compare different models easily:

```python
models = ['random_forest', 'svm', 'logistic_regression']
results = {}

for model_type in models:
    try:
        result = orchestrator.train(model_type)
        results[model_type] = result['evaluation']['roc_auc']
    except Exception as e:
        results[model_type] = f"Error: {e}"

# Print comparison
for model, score in results.items():
    print(f"{model}: {score}")
```

## Adding New Models

To add a new model type:

1. Create a new class inheriting from `ModelBase`:

```python
class MyCustomModel(ModelBase):
    def create_model(self):
        return MyMLAlgorithm(**self.get_model_params())
    
    def get_model_params(self) -> Dict[str, Any]:
        return self.config.get('model', {}).get('my_custom', {})
    
    def needs_scaling(self) -> bool:
        return True  # or False
```

2. Register it in the ModelFactory:

```python
ModelFactory._BASE_MODELS['my_custom'] = MyCustomModel
```

3. Add configuration section:

```json
{
  "model": {
    "my_custom": {
      "param1": "value1",
      "param2": "value2"
    }
  }
}
```

## Error Handling

The pipeline includes comprehensive error handling:

- **Model Loading**: Validates model files and compatibility
- **Data Validation**: Checks for required columns and data types
- **Feature Processing**: Handles missing values and invalid data
- **Training**: Catches and reports training failures
- **Detection**: Graceful handling of prediction errors

## Logging

Logging is configured automatically:

- Daily log files in `running_log/`
- Syslog integration for alerts
- Structured logging for all operations
- Debug mode available with `--verbose`

## Performance Considerations

- **Random Forest**: Fast training, good for large datasets
- **SVM**: Slower on large datasets, requires feature scaling
- **XGBoost**: Good performance/speed tradeoff, handles missing values
- **Feature Scaling**: Applied automatically when needed
- **Memory Usage**: Models are loaded on-demand

## Dependencies

Core requirements:
- pandas >= 1.1.5
- numpy >= 1.19.5
- scikit-learn >= 0.24.2
- joblib >= 1.0.1

Optional:
- xgboost (for XGBoost model)
- pymongo (for MongoDB data source)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Model Loading**: Check model file paths and permissions
3. **Memory Issues**: Use smaller datasets or reduce model complexity
4. **Feature Mismatch**: Ensure training and detection use same feature set

### Debug Mode

Enable verbose logging:

```bash
python pipeline/pipe.py train --model-type svm --verbose
```

### Model Information

Check available models and their requirements:

```python
from pipeline.pipe import ModelFactory
info = ModelFactory.get_model_info()
for model_type, details in info.items():
    print(f"{model_type}: scaling={details['requires_scaling']}")
```

## Migration from Legacy Pipeline

The OOP pipeline is designed to be compatible with existing data and configurations:

1. **Data Format**: Uses same input format as legacy pipeline
2. **Features**: Same feature engineering process
3. **Configuration**: Extends existing config structure
4. **Models**: Can load and use existing Random Forest models

To migrate:

1. Update imports: `from pipeline.pipe import PipelineOrchestrator`
2. Replace function calls with orchestrator methods
3. Update configuration file format (optional)
4. Test with existing data

## Future Enhancements

Planned improvements:
- Deep learning models (Neural Networks)
- Automated model selection
- Real-time model updating
- Distributed training support
- Web API interface