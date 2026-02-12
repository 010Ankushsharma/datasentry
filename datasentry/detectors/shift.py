import numpy as np
from typing import Dict
from .._utils import to_numpy


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Compute Population Stability Index.
    """

    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    psi = np.sum(
        (expected_perc - actual_perc)
        * np.log((expected_perc + 1e-12) / (actual_perc + 1e-12))
    )

    return float(psi)


def detect(X_train, X_test, threshold: float) -> Dict[str, float]:

    if X_test is None:
        return {"psi": 0.0, "is_problematic": False, "severity": 0.0}

    X_train = to_numpy(X_train)
    X_test = to_numpy(X_test)

    psi_values = [
        _psi(X_train[:, i], X_test[:, i])
        for i in range(X_train.shape[1])
    ]

    mean_psi = float(np.mean(psi_values))

    return {
        "psi": mean_psi,
        "is_problematic": bool(mean_psi > threshold),
        "severity": mean_psi,
    }
