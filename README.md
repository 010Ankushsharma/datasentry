# DataSentry

[![PyPI version](https://badge.fury.io/py/datasentry.svg)](https://badge.fury.io/py/datasentry)
[![Python Versions](https://img.shields.io/pypi/pyversions/datasentry.svg)](https://pypi.org/project/datasentry/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/010Ankushsharma/datasentry/workflows/CI/badge.svg)](https://github.com/010Ankushsharma/datasentry/actions)
[![codecov](https://codecov.io/gh/010Ankushsharma/datasentry/branch/main/graph/badge.svg)](https://codecov.io/gh/010Ankushsharma/datasentry)

A Professional Data Quality Framework for ML Pipelines

## Overview

DataSentry is a comprehensive Python library designed to detect and remediate data quality issues in machine learning pipelines. It provides a unified interface for identifying and fixing common data problems including class imbalance, label noise, data leakage, missing values, outliers, feature redundancy, and data distribution shift.

## Features

### Data Quality Detection
- **Imbalance Detection**: Identify class imbalance with customizable thresholds
- **Label Noise Detection**: Find potentially mislabeled samples using confident learning
- **Data Leakage Detection**: Detect target leakage, duplicates, and train-test contamination
- **Missing Value Detection**: Analyze missing value patterns and completeness
- **Outlier Detection**: Identify outliers using IQR, Z-score, Isolation Forest, and LOF
- **Redundancy Detection**: Find correlated and duplicate features
- **Shift Detection**: Detect distribution drift between train and test sets

### Data Quality Remediation
- **Imbalance Fixer**: SMOTE, ADASYN, undersampling, and class weights
- **Label Noise Fixer**: Remove, relabel, or weight noisy samples
- **Data Leakage Fixer**: Remove leaky features and duplicates
- **Missing Value Fixer**: Mean, median, mode, KNN, and iterative imputation
- **Outlier Fixer**: Remove, cap, transform, or winsorize outliers
- **Redundancy Fixer**: Remove features or apply PCA
- **Shift Fixer**: Standardize and normalize distributions

### Visualization
- Interactive plots for all data quality issues
- Distribution comparisons
- Correlation heatmaps
- Missing value patterns
- Outlier visualizations

## Installation

### From PyPI (Recommended)

```bash
pip install datasentry
```

### With Optional Dependencies

```bash
# For advanced imbalance handling (SMOTE, ADASYN)
pip install datasentry[imblearn]

# For all optional features
pip install datasentry[all]

# For development
pip install datasentry[dev]
```

### From Source

```bash
git clone https://github.com/010Ankushsharma/datasentry.git
cd datasentry
pip install -e .
```

## Quick Start

```python
from datasentry import DataSentry
import numpy as np

# Generate sample data
np.random.seed(42)
X_train = np.random.randn(1000, 10)
y_train = np.random.randint(0, 3, 1000)
X_test = np.random.randn(200, 10)

# Initialize DataSentry
ds = DataSentry(random_state=42, verbose=True)

# Generate comprehensive report
report = ds.generate_full_report(X_train, y_train, X_test=X_test)

# View health score
print(f"Health Score: {report['report_metadata']['health_score']:.2%}")
print(f"Overall Status: {report['report_metadata']['overall_status']}")

# Fix all detected issues
X_clean, y_clean = ds.fix_all(
    X_train, y_train,
    fix_config={
        'missing_values': {'strategy': 'mean'},
        'outliers': {'method': 'cap'},
        'imbalance': {'method': 'smote'},
    }
)
```

## Detailed Usage

### Individual Detection

```python
from datasentry import DataSentry

ds = DataSentry()

# Detect specific issues
imbalance_result = ds.detect_imbalance(X, y)
missing_result = ds.detect_missing_values(X)
outlier_result = ds.detect_outliers(X)
leakage_result = ds.detect_data_leakage(X, y, X_test=X_test)

# Check if issues were detected
if imbalance_result.issue_detected:
    print(f"Imbalance ratio: {imbalance_result.details['imbalance_ratio']}")
    print(f"Severity: {imbalance_result.severity}")
```

### Individual Fixing

```python
# Fix specific issues
from datasentry import MissingValueFixer, OutlierFixer

# Fix missing values
missing_fixer = MissingValueFixer(strategy='knn')
result = missing_fixer.fix(X, y)
X_fixed = result.X_transformed

# Fix outliers
outlier_fixer = OutlierFixer(method='winsorize')
result = outlier_fixer.fix(X, y)
X_fixed = result.X_transformed
```

### Visualization

```python
# Visualize data quality issues
import matplotlib.pyplot as plt

# Class imbalance
fig = ds.visualize_imbalance(y, plot_type='both')
plt.show()

# Missing values
fig = ds.visualize_missing_values(X, plot_type='matrix')
plt.show()

# Outliers
fig = ds.visualize_outliers(X, plot_type='box')
plt.show()

# Correlation heatmap
fig = ds.visualize_redundancy(X, plot_type='heatmap')
plt.show()

# Distribution shift
fig = ds.visualize_shift(X_train, X_test, plot_type='comparison')
plt.show()
```

### Report Generation

```python
# Generate HTML report
report_gen = ds.generate_full_report(X, y, X_test=X_test)

# Save as HTML
from datasentry.core.report import ReportGenerator

detectors = ds.detect_all(X, y, X_test=X_test)
report_gen = ReportGenerator(list(detectors.values()))
report_gen.save_html('data_quality_report.html')

# Save as JSON
report_gen.save_json('data_quality_report.json')
```

## API Reference

### Main Classes

#### `DataSentry`
Main orchestrator class for data quality management.

```python
DataSentry(
    random_state: int = 42,
    verbose: bool = True
)
```

**Methods:**
- `detect_all(X, y, X_test, y_test)` - Run all detectors
- `detect_imbalance(X, y)` - Detect class imbalance
- `detect_label_noise(X, y)` - Detect label noise
- `detect_data_leakage(X, y, X_test)` - Detect data leakage
- `detect_missing_values(X, y)` - Detect missing values
- `detect_outliers(X, y)` - Detect outliers
- `detect_redundancy(X, y)` - Detect feature redundancy
- `detect_shift(X, y, X_test, y_test)` - Detect distribution shift
- `fix_all(X, y, X_test, fix_config)` - Fix all issues
- `generate_full_report(X, y, X_test, y_test)` - Generate comprehensive report
- `visualize_*` - Various visualization methods

### Detectors

All detectors inherit from `BaseDetector` and return a `DetectionResult`.

```python
from datasentry import (
    ImbalanceDetector,
    LabelNoiseDetector,
    DataLeakageDetector,
    MissingValueDetector,
    OutlierDetector,
    RedundancyDetector,
    ShiftDetector,
)
```

### Fixers

All fixers inherit from `BaseFixer` and return a `FixResult`.

```python
from datasentry import (
    ImbalanceFixer,
    LabelNoiseFixer,
    DataLeakageFixer,
    MissingValueFixer,
    OutlierFixer,
    RedundancyFixer,
    ShiftFixer,
)
```

## Examples

See the `examples/` directory for more detailed examples:

- `basic_example.py` - Basic usage of DataSentry
- `advanced_example.py` - Advanced features and customization
- `pipeline_integration.py` - Integration with sklearn pipelines

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Support

- **Documentation**: [https://datasentry.readthedocs.io](https://datasentry.readthedocs.io)
- **Issue Tracker**: [GitHub Issues](https://github.com/010Ankushsharma/datasentry/issues)
- **Discussions**: [GitHub Discussions](https://github.com/010Ankushsharma/datasentry/discussions)

## Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/)
- Inspired by data quality best practices in ML pipelines
