import numpy as np
from typing import Dict
from .._utils import validate_y


def detect(y: np.ndarray, threshold: float = 3.0) -> Dict[str, float]:
    """
    Detect class imbalance using normalized majority/minority ratio.

    imbalance_score is scaled between 0 and 1:
        0   → perfectly balanced
        ~1  → highly imbalanced
    """

    y = validate_y(y)

    _, counts = np.unique(y, return_counts=True)

    if len(counts) < 2:
        ratio = float("inf")
        imbalance_score = 1.0
    else:
        max_count = np.max(counts)
        min_count = np.min(counts)

        if min_count == 0:
            ratio = float("inf")
            imbalance_score = 1.0
        else:
            ratio = max_count / min_count
            imbalance_score = (ratio - 1) / ratio  # normalized 0–1

    return {
        "imbalance_score": float(imbalance_score),
        "is_problematic": bool(ratio > threshold),
        "severity": float(imbalance_score),
    }
