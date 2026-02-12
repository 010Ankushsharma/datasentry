DataSentry

Proactive dataset validation for reliable machine learning systems.

DataSentry is a production-focused Python library designed to detect structural dataset issues before model training begins. It provides automated diagnostics to help prevent silent performance degradation, misleading evaluation metrics, and costly deployment failures.

Modern ML systems fail more often due to data problems than model architecture. DataSentry ensures your data is trustworthy before your model goes to production.

Why DataSentry?

Machine learning pipelines commonly suffer from:

Severe class imbalance

Noisy or inconsistent labels

Hidden data leakage

Extreme outliers

Distribution shift between datasets

These issues often go unnoticed until late in development — or worse, after deployment.

DataSentry provides structured, automated checks to identify these risks early.

Core Capabilities
Class Imbalance Detection

Evaluates class distribution and computes imbalance metrics to detect biased datasets.

Label Noise Detection

Identifies suspicious label patterns that may reduce model generalization.

Data Leakage Detection

Flags features highly correlated with the target variable to prevent inflated validation performance.

Outlier Detection

Detects abnormal samples that may distort training behavior and model stability.

Distribution Shift Detection

Evaluates feature distribution differences to identify drift or dataset mismatch.

Installation

Install from PyPI:

pip install datasentry


Install from source:

pip install .

Requirements

Python 3.9+

numpy

pandas

Quick Start
import numpy as np
from datasentry import analyze

# Example dataset
X = np.random.randn(100, 5)
y = np.array([0] * 90 + [1] * 10)

report = analyze(X=X, y=y)

print(report)


Example structured output:

{
    "imbalance": {
        "imbalance_score": 9.0,
        "is_imbalanced": True
    },
    "outliers": {
        "outlier_fraction": 0.05
    },
    "label_noise": {
        "noise_score": 0.08
    },
    "leakage": {
        "leakage_detected": False
    },
    "shift": {
        "shift_score": 0.02
    }
}

Advanced Usage

Custom thresholds can be configured to control sensitivity:

report = analyze(
    X=X,
    y=y,
    imbalance_threshold=3.0,
    outlier_threshold=0.1,
    leakage_threshold=0.9
)


This enables integration into:

CI pipelines

Pre-training validation steps

Automated ML workflows

Data quality monitoring systems

Architecture
datasentry/
│
├── detectors/
│   ├── imbalance.py
│   ├── label_noise.py
│   ├── leakage.py
│   ├── outliers.py
│   └── shift.py
│
├── analyzer.py
├── config.py
├── report.py
├── utils.py
└── fixer.py


Design principles:

Modular detector-based design

Clear separation of concerns

Structured and extensible reporting

Test-covered components

CI-enabled development workflow

Running Tests
pytest

Development Setup
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -e .
pip install pytest

Roadmap

Command-line interface (CLI)

Visualization utilities

Enhanced statistical leakage detection

Automated remediation suggestions

scikit-learn pipeline integration

Production drift monitoring

Contributing

Contributions are welcome. Please ensure:

Clear documentation

Unit test coverage

Consistent code standards

Descriptive commit messages

License

MIT License

Philosophy

Reliable machine learning begins with reliable data.

DataSentry focuses on structural dataset validation to reduce downstream debugging effort, improve model robustness, and increase production reliability.
