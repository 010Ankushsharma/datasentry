from typing import Optional
from .config import DataSentryConfig
from .detectors import imbalance, outliers, shift, leakage, label_noise
from .report import Report
from .fixer import AutoFixer


def analyze(X, y, X_test=None, config: Optional[DataSentryConfig] = None) -> Report:
    """
    Run full data inspection pipeline.

    Parameters
    ----------
    X : array-like
        Training features
    y : array-like
        Target labels
    X_test : array-like, optional
        Test features for drift detection
    config : DataSentryConfig
        Custom configuration

    Returns
    -------
    Report
    """

    config = config or DataSentryConfig()

    issues = {
        "imbalance": imbalance.detect(y, config.imbalance_threshold),
        "outliers": outliers.detect(X, config.outlier_contamination, config.random_state),
        "distribution_shift": shift.detect(X, X_test, config.drift_threshold),
        "data_leakage": leakage.detect(X, y, config.leakage_threshold),
        "label_noise": label_noise.detect(X, y, config.noise_threshold, config.random_state),
    }

    report = Report(issues)
    report.fixer = AutoFixer(issues, config)

    return report
