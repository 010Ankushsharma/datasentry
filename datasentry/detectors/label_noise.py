# datasentry/detectors/label_noise.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .._utils import validate_X, validate_y

def detect(X, y, threshold: float = 0.8) -> dict:

    X = validate_X(X)
    y = validate_y(y)

    if len(np.unique(y)) < 2:
        return {
            "status": "skipped",
            "reason": "Single class only",
            "noise_detected": False
        }

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    probs = model.predict_proba(X)
    preds = model.predict(X)

    disagreement = preds != y
    high_conf_wrong = [
        i for i, wrong in enumerate(disagreement)
        if wrong and np.max(probs[i]) > threshold
    ]

    noise_ratio = len(high_conf_wrong) / len(y)

    return {
        "status": "warning" if noise_ratio > 0.05 else "ok",
        "num_suspected_noisy_labels": len(high_conf_wrong),
        "noise_ratio": float(noise_ratio),
        "noise_detected": bool(noise_ratio > 0.05)
    }
