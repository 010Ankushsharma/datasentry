# DataSentry

> A production-oriented Python library for detecting structural dataset issues before machine learning model training.

DataSentry is a lightweight and extensible data validation framework designed to identify critical data problems early in the ML lifecycle. It helps prevent silent performance degradation, misleading validation metrics, and deployment failures caused by poor dataset quality.

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Features](#features)
- [Installation](#installation)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Running Tests](#running-tests)
- [Development Setup](#development-setup)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

DataSentry provides automated dataset diagnostics to ensure data reliability before model training begins.

It detects:

- Class imbalance
- Label noise
- Data leakage
- Outliers
- Distribution shift

The library is model-agnostic and can be integrated into any ML workflow.

---

## Motivation

Machine learning systems often fail due to data quality issues rather than model complexity. Common failure scenarios include:

- Severe class imbalance causing biased predictions
- Noisy or inconsistent labels reducing generalization
- Data leakage inflating validation accuracy
- Outliers distorting feature distributions
- Distribution drift breaking deployed models

DataSentry addresses these problems proactively through automated checks.

---

## Features

### 1. Class Imbalance Detection
Evaluates class distribution and computes imbalance scores.

### 2. Label Noise Detection
Identifies irregular label patterns that may indicate annotation errors.

### 3. Data Leakage Detection
Flags suspicious correlations between features and target variables.

### 4. Outlier Detection
Detects abnormal samples using statistical methods.

### 5. Distribution Shift Detection
Compares feature distributions to identify drift or mismatch.

---

## Installation

Install from PyPI:

```bash
pip install datasentry

```Install locally:
pip install .

### Requirements

Python 3.9+
numpy
pandas

### Quick Start

import numpy as np
from datasentry import analyze

# Example dataset
X = np.random.randn(100, 5)
y = np.array([0] * 90 + [1] * 10)

report = analyze(X=X, y=y)

print(report)


```Example output:
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

### Detailed Usage
You can customize analysis thresholds using configuration parameters:

from datasentry import analyze

report = analyze(
    X=X,
    y=y,
    imbalance_threshold=3.0,
    outlier_threshold=0.1,
    leakage_threshold=0.9
)

Each detector returns structured diagnostic metrics that can be used for:

Automated pipeline validation
CI-based dataset checks
Pre-training quality gates
Data monitoring workflows
