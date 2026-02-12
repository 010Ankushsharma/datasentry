from typing import Dict
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from .._utils import to_numpy, validate_y


def detect(X, y, threshold: float) -> Dict[str, float]:
    """
    Detect potential feature leakage via mutual information.
    """

    X = to_numpy(X)
    y = validate_y(y)

    mi_scores = mutual_info_classif(X, y)
    max_mi = float(np.max(mi_scores))

    return {
        "max_mutual_information": max_mi,
        "is_problematic": bool(max_mi > threshold),
        "severity": max_mi,
    }
