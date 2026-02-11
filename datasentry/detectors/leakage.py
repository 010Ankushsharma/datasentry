# datasentry/detectors/leakage.py
import numpy as np
from .._utils import validate_X, validate_y

def detect(X, y, threshold: float = 0.95) -> dict:

    X = validate_X(X)
    y = validate_y(y)

    if not np.issubdtype(y.dtype, np.number):
        return {
            "status": "skipped",
            "reason": "Non-numeric target",
            "leakage_detected": False
        }

    correlations = []

    for i in range(X.shape[1]):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append(abs(corr))

    correlations = np.nan_to_num(correlations)
    suspicious = np.array(correlations) > threshold

    return {
        "status": "warning" if suspicious.any() else "ok",
        "num_suspicious_features": int(suspicious.sum()),
        "leakage_detected": bool(suspicious.any())
    }
