# datasentry/detectors/shift.py
import numpy as np
from scipy.stats import ks_2samp
from .._utils import validate_X

def detect(X_train, X_test, alpha: float = 0.05) -> dict:

    if X_test is None:
        return {
            "status": "skipped",
            "reason": "X_test not provided",
            "shift_detected": False
        }

    X_train = validate_X(X_train)
    X_test = validate_X(X_test)

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train and test must have same number of features.")

    shifted = []

    for i in range(X_train.shape[1]):
        _, p_value = ks_2samp(X_train[:, i], X_test[:, i])
        shifted.append(p_value < alpha)

    shifted = np.array(shifted)

    return {
        "status": "warning" if shifted.any() else "ok",
        "num_shifted_features": int(shifted.sum()),
        "shift_detected": bool(shifted.any())
    }
