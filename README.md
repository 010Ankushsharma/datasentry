# 🚀 DataSentry

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

> Proactive dataset validation for reliable machine learning systems.

DataSentry is a lightweight, production-focused Python library that
detects structural dataset issues **before model training begins**.\
It helps prevent silent performance degradation, misleading validation
metrics, and costly deployment failures.

Because in machine learning --- **bad data breaks good models.**

------------------------------------------------------------------------

## 🚩 Why DataSentry?

Modern ML pipelines frequently suffer from hidden dataset issues:

  Problem                 Risk
  ----------------------- ------------------------------
  ⚖️ Class Imbalance      Biased predictions
  🏷 Label Noise           Reduced generalization
  🔓 Data Leakage         Inflated validation accuracy
  📉 Outliers             Distorted feature space
  🔄 Distribution Shift   Poor real-world performance

DataSentry automatically identifies these risks early in your workflow.

------------------------------------------------------------------------

## 🧠 Core Features

### ⚖️ Class Imbalance Detection

Evaluates label distribution and computes normalized imbalance metrics.

### 🏷 Label Noise Detection

Flags suspicious label inconsistencies affecting model learning.

### 🔓 Data Leakage Detection

Identifies features overly correlated with target variables.

### 📉 Outlier Detection

Detects anomalous samples that may distort training.

### 🔄 Distribution Shift Detection

Compares feature distributions to detect dataset drift.

------------------------------------------------------------------------

## 📦 Installation

Install from PyPI:

``` bash
pip install datasentry
```

Or install locally:

``` bash
pip install .
```

------------------------------------------------------------------------

## ⚡ Quick Start

``` python
import numpy as np
from datasentry import analyze

X = np.random.randn(100, 5)
y = np.array([0] * 90 + [1] * 10)

report = analyze(X=X, y=y)
print(report)
```

------------------------------------------------------------------------

## ⚙ Advanced Usage

``` python
report = analyze(
    X=X,
    y=y,
    imbalance_threshold=3.0,
    outlier_threshold=0.1,
    leakage_threshold=0.9
)
```

------------------------------------------------------------------------

## 🏗 Architecture

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

------------------------------------------------------------------------

## 🧪 Running Tests

``` bash
pytest
```

------------------------------------------------------------------------

## 🗺 Roadmap

-   CLI interface\
-   Visualization dashboard\
-   Advanced statistical leakage detection\
-   Automated remediation suggestions\
-   scikit-learn pipeline integration\
-   Production drift monitoring

------------------------------------------------------------------------

## 🤝 Contributing

Contributions are welcome. Please ensure:

-   Clear documentation\
-   Proper unit test coverage\
-   Consistent coding standards\
-   Descriptive commit messages

------------------------------------------------------------------------

## 📄 License

MIT License

------------------------------------------------------------------------

## 💡 Philosophy

Reliable machine learning begins with reliable data.

DataSentry focuses on structural dataset validation to reduce debugging
effort, improve model robustness, and increase production stability.
