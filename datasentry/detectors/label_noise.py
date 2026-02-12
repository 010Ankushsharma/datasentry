from typing import Dict
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from .._utils import to_numpy, validate_y


def detect(X, y, threshold: float, random_state: int) -> Dict[str, float]:
    """
    Detect label noise using cross-validated disagreement.
    """

    X = to_numpy(X)
    y = validate_y(y)

    if len(np.unique(y)) < 2:
        return {"noise_ratio": 0.0, "is_problematic": False, "severity": 0.0}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    preds = np.zeros(len(y))

    for train_idx, val_idx in skf.split(X, y):
        clf = RandomForestClassifier(random_state=random_state)
        clf.fit(X[train_idx], y[train_idx])
        preds[val_idx] = clf.predict(X[val_idx])

    noise_ratio = float((preds != y).mean())

    return {
        "noise_ratio": noise_ratio,
        "is_problematic": bool(noise_ratio > threshold),
        "severity": noise_ratio,
    }
