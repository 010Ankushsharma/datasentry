# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-22

### Added
- Initial release of DataSentry
- **Detectors**: 7 data quality detectors
  - `ImbalanceDetector`: Detect class imbalance in classification datasets
  - `LabelNoiseDetector`: Identify potentially mislabeled samples
  - `DataLeakageDetector`: Find data leakage issues including target leakage and duplicates
  - `MissingValueDetector`: Analyze missing value patterns
  - `OutlierDetector`: Detect outliers using IQR, Z-score, Isolation Forest, and LOF
  - `RedundancyDetector`: Identify correlated and duplicate features
  - `ShiftDetector`: Detect distribution shift between train and test sets
- **Fixers**: 7 data quality fixers
  - `ImbalanceFixer`: Fix imbalance with SMOTE, ADASYN, undersampling, and class weights
  - `LabelNoiseFixer`: Clean label noise by removal, relabeling, or weighting
  - `DataLeakageFixer`: Remove leaky features and duplicates
  - `MissingValueFixer`: Impute missing values with various strategies
  - `OutlierFixer`: Handle outliers by removal, capping, transformation, or winsorization
  - `RedundancyFixer`: Remove redundant features or apply PCA
  - `ShiftFixer`: Standardize data to reduce distribution shift
- **Visualizers**: 7 visualization modules
  - `ImbalanceVisualizer`: Bar charts, pie charts, and comparison plots
  - `NoiseVisualizer`: Confusion matrices and noise score distributions
  - `LeakageVisualizer`: Correlation heatmaps and feature importance plots
  - `MissingVisualizer`: Missingness matrix and bar charts
  - `OutlierVisualizer`: Box plots, scatter plots, and distribution plots
  - `RedundancyVisualizer`: Correlation heatmaps and pair plots
  - `ShiftVisualizer`: Distribution comparisons and Q-Q plots
- **Core Components**
  - `DataSentry`: Main orchestrator class for unified API
  - `ReportGenerator`: Generate HTML and JSON reports
  - `DetectionResult` and `FixResult`: Standardized result containers
  - `SeverityLevel`: Enum for issue severity classification
- **Documentation**
  - Comprehensive README with usage examples
  - Contributing guidelines
  - MIT License
- **CI/CD**
  - GitHub Actions workflow for testing and publishing
  - Support for Python 3.9, 3.10, 3.11, and 3.12
  - Automated PyPI publishing on version tags

### Features
- Unified API for detection, fixing, and visualization
- Individual function calls for each data quality issue
- `detect_all()` method to run all detectors at once
- `fix_all()` method to fix all issues with configurable strategies
- `generate_full_report()` for comprehensive data quality reports
- HTML and JSON report export capabilities
- Type hints throughout the codebase
- Comprehensive docstrings with examples

[1.0.0]: https://github.com/010Ankushsharma/datasentry/releases/tag/v1.0.0
