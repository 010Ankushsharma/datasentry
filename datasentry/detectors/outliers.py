# datasentry/detectors/outliers.py
import numpy as np
from .._utils import validate_X

def detect(X, z_threshold: float = 3.0) -> dict:

    X = validate_X(X)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1e-8

    z_scores = np.abs((X - mean) / std)
    outlier_mask = (z_scores > z_threshold).any(axis=1)

    ratio = float(outlier_mask.mean())

    return {
        "status": "warning" if ratio > 0 else "ok",
        "num_outliers": int(outlier_mask.sum()),
        "outlier_ratio": ratio,
        "is_problematic": bool(ratio > 0)
    }
