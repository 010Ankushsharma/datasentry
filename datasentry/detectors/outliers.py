from typing import Dict
from sklearn.ensemble import IsolationForest
from .._utils import to_numpy


def detect(X, contamination: float, random_state: int) -> Dict[str, float]:
    """
    Detect outliers using Isolation Forest.
    """

    X = to_numpy(X)

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
    )

    preds = model.fit_predict(X)
    outlier_ratio = (preds == -1).mean()

    return {
        "outlier_ratio": float(outlier_ratio),
        "is_problematic": bool(outlier_ratio > contamination),
        "severity": float(outlier_ratio),
    }
