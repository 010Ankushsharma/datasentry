# datasentry/analyzer.py
from .detectors import imbalance, outliers, shift, leakage, label_noise
from .fixer import AutoFixer
from .report import Report

def analyze(X, y, X_test=None):

    issues = {
        "imbalance": imbalance.detect(y),
        "outliers": outliers.detect(X),
        "distribution_shift": shift.detect(X, X_test),
        "data_leakage": leakage.detect(X, y),
        "label_noise": label_noise.detect(X, y)
    }

    report = Report(issues)
    report.fixer = AutoFixer(issues)

    return report
