# datasentry/detectors/imbalance.py
import numpy as np
from .._utils import validate_y

def detect(y, threshold: float = 3.0) -> dict:

    y = validate_y(y)

    classes, counts = np.unique(y, return_counts=True)

    if len(counts) < 2:
        return {
            "status": "ok",
            "reason": "Single class only",
            "imbalance_ratio": 1.0,
            "is_imbalanced": False
        }

    min_count = counts.min()

    if min_count == 0:
        return {
            "status": "warning",
            "reason": "Zero-sample class detected",
            "imbalance_ratio": float("inf"),
            "is_imbalanced": True
        }

    ratio = counts.max() / min_count

    return {
        "status": "warning" if ratio > threshold else "ok",
        "imbalance_ratio": float(ratio),
        "classes": classes.tolist(),
        "counts": counts.tolist(),
        "is_imbalanced": bool(ratio > threshold)
    }
